#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/Detector.h"
#include "DD4hep/Shapes.h"
#include <TGeoBBox.h>
#include <TGeoBoolNode.h>
#include <vector>
#include <charconv>

using namespace cms;
using namespace edm;
using namespace std;
using namespace cms::dd;

dd4hep::Solid DDSolid::solidA() const {
  if (dd4hep::isA<dd4hep::SubtractionSolid>(solid_) or dd4hep::isA<dd4hep::UnionSolid>(solid_) or
      dd4hep::isA<dd4hep::IntersectionSolid>(solid_)) {
    const TGeoCompositeShape* sh = (const TGeoCompositeShape*)solid_.ptr();
    const TGeoBoolNode* boolean = sh->GetBoolNode();
    TGeoShape* solidA = boolean->GetLeftShape();
    return dd4hep::Solid(solidA);
  }
  return solid_;
}

dd4hep::Solid DDSolid::solidB() const {
  if (dd4hep::isA<dd4hep::SubtractionSolid>(solid_) or dd4hep::isA<dd4hep::UnionSolid>(solid_) or
      dd4hep::isA<dd4hep::IntersectionSolid>(solid_)) {
    const TGeoCompositeShape* sh = static_cast<const TGeoCompositeShape*>(solid_.ptr());
    const TGeoBoolNode* boolean = sh->GetBoolNode();
    TGeoShape* solidB = boolean->GetRightShape();
    return dd4hep::Solid(solidB);
  }
  return solid_;
}

const std::vector<double> DDSolid::parameters() const { return solid().dimensions(); }

DDFilteredView::DDFilteredView(const DDDetector* det, const Volume volume) : registry_(&det->specpars()) {
  it_.emplace_back(Iterator(volume));
}

DDFilteredView::DDFilteredView(const DDCompactView& cpv, const cms::DDFilter& filter) : registry_(&cpv.specpars()) {
  it_.emplace_back(Iterator(cpv.detector()->worldVolume()));
  registry_->filter(refs_, filter.attribute(), filter.value());
  mergedSpecifics(refs_);
  LogVerbatim("Geometry").log([&](auto& log) {
    log << "Filtered by an attribute " << filter.attribute() << "==" << filter.value()
        << " DD SpecPar Registry size: " << refs_.size() << "\n";
    for (const auto& t : refs_) {
      log << "\nRegExps { ";
      for (const auto& ki : t->paths)
        log << ki << " ";
      log << "};\n ";
      for (const auto& kl : t->spars) {
        log << kl.first << " = ";
        for (const auto& kil : kl.second) {
          log << kil << " ";
        }
        log << "\n ";
      }
    }
  });
}

const PlacedVolume DDFilteredView::volume() const {
  assert(node_);
  return PlacedVolume(node_);
}

//
// This should be used for debug purpose only
//
const std::string DDFilteredView::path() const {
  if (it_.empty()) {
    return std::string();
  }
  TString fullPath;
  it_.back().GetPath(fullPath);
  return std::string(fullPath.Data());
}

//
// The vector is filled from bottom up:
// result[0] contains the current node copy number
//
const std::vector<int> DDFilteredView::copyNos() const {
  std::vector<int> result;

  if (not it_.empty()) {
    for (int i = it_.back().GetLevel(); i > 0; --i) {
      result.emplace_back(it_.back().GetNode(i)->GetNumber());
    }
  }

  return result;
}

const Double_t* DDFilteredView::trans() const { return it_.back().GetCurrentMatrix()->GetTranslation(); }

const Translation DDFilteredView::translation() const {
  const Double_t* translation = it_.back().GetCurrentMatrix()->GetTranslation();
  assert(translation);
  return Translation(translation[0], translation[1], translation[2]);
}

const Translation DDFilteredView::translation(const std::vector<Node*>& nodes) const {
  const TGeoMatrix* current = it_.back().GetCurrentMatrix();
  TGeoHMatrix matrix(*current);
  for (const auto& n : nodes) {
    matrix.Multiply(n->GetMatrix());
  }
  const Double_t* translation = matrix.GetTranslation();
  assert(translation);
  return Translation(translation[0], translation[1], translation[2]);
}

const Double_t* DDFilteredView::rot() const { return it_.back().GetCurrentMatrix()->GetRotationMatrix(); }

const RotationMatrix DDFilteredView::rotation() const {
  const Double_t* rotation = it_.back().GetCurrentMatrix()->GetRotationMatrix();
  if (rotation == nullptr) {
    LogError("DDFilteredView") << "Current node has no valid rotation matrix.";
    return RotationMatrix();
  }

  RotationMatrix rotMatrix;
  rotMatrix.SetComponents(rotation[0],
                          rotation[1],
                          rotation[2],
                          rotation[3],
                          rotation[4],
                          rotation[5],
                          rotation[6],
                          rotation[7],
                          rotation[8]);
  return rotMatrix;
}

void DDFilteredView::rot(dd4hep::Rotation3D& matrixOut) const {
  const Double_t* rotation = it_.back().GetCurrentMatrix()->GetRotationMatrix();
  if (rotation == nullptr) {
    LogError("DDFilteredView") << "Current node has no valid rotation matrix.";
    return;
  }
  matrixOut.SetComponents(rotation[0],
                          rotation[1],
                          rotation[2],
                          rotation[3],
                          rotation[4],
                          rotation[5],
                          rotation[6],
                          rotation[7],
                          rotation[8]);
}

void DDFilteredView::mergedSpecifics(DDSpecParRefs const& specs) {
  if (!filters_.empty()) {
    filters_.clear();
    filters_.shrink_to_fit();
  }
  for (const auto& i : specs) {
    for (const auto& j : i->paths) {
      vector<string_view> toks = split(j, "/");
      auto const& filter = find_if(begin(filters_), end(filters_), [&](auto const& f) {
        auto const& k = find_if(begin(f->skeys), end(f->skeys), [&](auto const& p) { return toks.front() == p; });
        if (k != end(f->skeys)) {
          currentFilter_ = f.get();
          return true;
        }
        return false;
      });
      if (filter == end(filters_)) {
        filters_.emplace_back(unique_ptr<Filter>(
            new Filter{{toks.front()},
                       {std::regex(std::string("^").append({toks.front().data(), toks.front().size()}).append("$"))},
                       nullptr,
                       nullptr,
                       i}));
        // initialize current filter if it's empty
        if (currentFilter_ == nullptr) {
          currentFilter_ = filters_.back().get();
        }
      }
      // all next levels
      for (size_t pos = 1; pos < toks.size(); ++pos) {
        if (currentFilter_->next != nullptr) {
          currentFilter_ = currentFilter_->next.get();
          auto const& l = find_if(begin(currentFilter_->skeys), end(currentFilter_->skeys), [&](auto const& p) {
            return toks.front() == p;
          });
          if (l == end(currentFilter_->skeys)) {
            currentFilter_->skeys.emplace_back(toks[pos]);
            currentFilter_->keys.emplace_back(
                std::regex(std::string("^").append({toks[pos].data(), toks[pos].size()}).append("$")));
          }
        } else {
          currentFilter_->next.reset(
              new Filter{{toks[pos]},
                         {std::regex(std::string("^").append({toks[pos].data(), toks[pos].size()}).append("$"))},
                         nullptr,
                         currentFilter_,
                         i});
        }
      }
    }
  }
}

void DDFilteredView::printFilter() const {
  for (const auto& f : filters_) {
    f->print();
  }
}

bool DDFilteredView::firstChild() {
  if (it_.empty()) {
    LogVerbatim("DDFilteredView") << "Iterator vector has zero size.";
    return false;
  }
  it_.back().SetType(0);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (accept(noNamespace(node->GetVolume()->GetName()))) {
      node_ = node;
      startLevel_ = it_.back().GetLevel();
      return true;
    }
  }
  LogVerbatim("DDFilteredView") << "Search for first child failed.";
  return false;
}

int DDFilteredView::nodeCopyNo(const std::string_view copyNo) const {
  int result;
  if (auto [p, ec] = std::from_chars(copyNo.data(), copyNo.data() + copyNo.size(), result); ec == std::errc()) {
    return result;
  }
  return -1;
}

std::vector<std::pair<std::string_view, int>> DDFilteredView::toNodeNames(const std::string& path) {
  std::vector<std::pair<std::string_view, int>> result;
  std::vector<string_view> names = split(path, "/");
  for (const auto& i : names) {
    auto name = noNamespace(i);
    int copyNo = -1;
    auto lpos = name.find_first_of('[');
    if (lpos != std::string::npos) {
      auto rpos = name.find_last_of(']');
      if (rpos != std::string::npos) {
        copyNo = nodeCopyNo(name.substr(lpos + 1, rpos - 1));
      }
      result.emplace_back(name.substr(0, lpos), copyNo);
    } else {
      result.emplace_back(name, -1);
    }
  }

  return result;
}

bool DDFilteredView::match(const std::string& path, const std::vector<std::pair<std::string_view, int>>& names) const {
  std::vector<std::pair<std::string_view, int>> toks;
  std::vector<string_view> pnames = split(path, "/");
  for (const auto& i : pnames) {
    auto name = noNamespace(i);
    auto lpos = name.find_first_of('_');
    if (lpos != std::string::npos) {
      int copyNo = nodeCopyNo(name.substr(lpos + 1));
      toks.emplace_back(name.substr(0, lpos), copyNo);
    }
  }
  if (toks.size() != names.size()) {
    return false;
  }

  for (unsigned int i = 0; i < names.size(); i++) {
    if (names[i].first != toks[i].first) {
      return false;
    } else {
      if (names[i].second != -1 and names[i].second != toks[i].second) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::vector<Node*>> DDFilteredView::children(const std::string& selectPath) {
  std::vector<std::vector<Node*>> paths;
  if (it_.empty()) {
    LogVerbatim("DDFilteredView") << "Iterator vector has zero size.";
    return paths;
  }
  if (node_ == nullptr) {
    throw cms::Exception("DDFilteredView") << "Can't get children of a null node. Please, call firstChild().";
  }
  it_.back().SetType(0);
  std::vector<std::pair<std::string_view, int>> names = toNodeNames(selectPath);
  auto rit = names.rbegin();
  Node* node = it_.back().Next();
  while (node != nullptr) {
    if (node->GetVolume()->GetName() == rit->first) {
      std::string pathToNode = path();
      std::string::size_type n = pathToNode.find(node_->GetName());
      std::string pathFromParent = pathToNode.substr(n);

      if (match(pathFromParent, names)) {
        std::vector<Node*> result;
        LogVerbatim("Geometry") << "Match found: " << pathFromParent;
        for (int i = startLevel_; i < it_.back().GetLevel(); i++) {
          result.emplace_back(it_.back().GetNode(i));
        }
        result.emplace_back(node);
        paths.emplace_back(result);
      }
    }
    node = it_.back().Next();
  }
  return paths;
}

bool DDFilteredView::firstSibling() {
  assert(node_);
  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  Node* node = nullptr;
  if (next(0) == false)
    return false;
  node = node_;
  it_.emplace_back(Iterator(it_.back()));
  it_.back().SetType(1);
  if (currentFilter_ != nullptr and currentFilter_->next != nullptr)
    currentFilter_ = currentFilter_->next.get();
  else
    return false;
  do {
    if (accepted(currentFilter_->keys, noNamespace(node->GetVolume()->GetName()))) {
      node_ = node;
      return true;
    }
  } while ((node = it_.back().Next()));

  return false;
}

bool DDFilteredView::nextSibling() {
  assert(node_);
  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  if (it_.back().GetType() == 0)
    return firstSibling();
  else {
    up();
    it_.back().SetType(1);
    Node* node = node_;
    do {
      if (accepted(currentFilter_->keys, noNamespace(node->GetVolume()->GetName()))) {
        node_ = node;
        return true;
      }
    } while ((node = it_.back().Next()));

    return false;
  }
}

bool DDFilteredView::sibling() {
  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (accepted(currentFilter_->keys, node->GetVolume()->GetName())) {
      node_ = node;
      return true;
    }
  }
  return false;
}

bool DDFilteredView::checkChild() {
  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (accepted(currentFilter_->keys, node->GetVolume()->GetName())) {
      return true;
    }
  }
  return false;
}

bool DDFilteredView::parent() {
  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  up();
  it_.back().SetType(0);

  return true;
}

bool DDFilteredView::next(int type) {
  if (it_.empty())
    return false;
  it_.back().SetType(type);
  Node* node = nullptr;
  if ((node = it_.back().Next())) {
    node_ = node;
    return true;
  } else
    return false;
}

void DDFilteredView::down() {
  if (it_.empty() or currentFilter_ == nullptr)
    return;
  it_.emplace_back(Iterator(it_.back()));
  next(0);
  if (currentFilter_->next)
    currentFilter_ = currentFilter_->next.get();
}

void DDFilteredView::up() {
  if (it_.size() > 1 and currentFilter_ != nullptr) {
    it_.pop_back();
    it_.back().SetType(0);
    if (currentFilter_->up)
      currentFilter_ = currentFilter_->up;
  }
}

bool DDFilteredView::accept(std::string_view name) {
  bool result = false;
  for (const auto& it : filters_) {
    currentFilter_ = it.get();
    result = accepted(currentFilter_->keys, name);
    if (result)
      return result;
  }
  return result;
}

const std::vector<double> DDFilteredView::parameters() const {
  assert(node_);
  std::cout << "DDFilteredView::parameters, after assert " << std::endl;
  Volume currVol = node_->GetVolume();
  // Boolean shapes are a special case
  if (currVol->GetShape()->IsA() == TGeoCompositeShape::Class()) {
    std::cout << "boolean shape " << std::endl;
    const TGeoCompositeShape* shape = static_cast<const TGeoCompositeShape*>(currVol->GetShape());
    const TGeoBoolNode* boolean = shape->GetBoolNode();
    std::cout << "reached boolean " << std::endl;
    while (boolean && boolean->GetLeftShape() && boolean->GetLeftShape()->IsA() != TGeoBBox::Class()) {
      std::cout << "before" << std::endl;
      boolean = static_cast<const TGeoCompositeShape*>(boolean->GetLeftShape())->GetBoolNode();
      std::cout << "after" << std::endl;
      std::cout << "TGeoCompositeShape::Class = " << (boolean->GetLeftShape()->IsA() == TGeoCompositeShape::Class()) << std::endl;
    }
    std::cout << "reached end while " << std::endl;
    const TGeoBBox* box = static_cast<const TGeoBBox*>(boolean->GetLeftShape());
    std::cout << "reached end box " << std::endl;
    return {box->GetDX(), box->GetDY(), box->GetDZ()};
  } else {
    std::cout << "currVol.solid().dimensions() " << std::endl;
    return currVol.solid().dimensions();
  }
}

const cms::DDSolidShape DDFilteredView::shape() const {
  return cms::dd::value(cms::DDSolidShapeMap, std::string(node_->GetVolume()->GetShape()->GetTitle()));
}

LegacySolidShape DDFilteredView::legacyShape(const cms::DDSolidShape shape) const {
  return cms::dd::value(cms::LegacySolidShapeMap, shape);
}

template <>
std::string_view DDFilteredView::get<string_view>(const string& key) const {
  std::string_view result;
  DDSpecParRefs refs;
  registry_->filter(refs, key, "");
  int level = it_.back().GetLevel();
  for_each(begin(refs), end(refs), [&](auto const& i) {
    auto k = find_if(begin(i->paths), end(i->paths), [&](auto const& j) {
      auto const& names = split(realTopName(j), "/");
      int count = names.size();
      bool flag = false;
      for (int nit = level; count > 0 and nit > 0; --nit) {
        std::string_view name = noNamespace(it_.back().GetNode(nit)->GetVolume()->GetName());
        if (!regex_match(std::string(name.data(), name.size()), regex(std::string(names[--count])))) {
          flag = false;
          break;
        } else {
          flag = true;
        }
      }
      return flag;
    });
    if (k != end(i->paths)) {
      result = i->strValue(key);
    }
  });

  return result;
}

template <>
double DDFilteredView::get<double>(const string& key) const {
  double result(0.0);
  std::string_view tmpStrV = get<std::string_view>(key);
  if (!tmpStrV.empty())
    result = dd4hep::_toDouble({tmpStrV.data(), tmpStrV.size()});
  return result;
}

template <>
std::vector<double> DDFilteredView::get<std::vector<double>>(const string& name, const string& key) const {
  if (registry_->hasSpecPar(name))
    return registry_->specPar(name)->value<std::vector<double>>(key);
  else
    return std::vector<double>();
}

template <>
std::vector<int> DDFilteredView::get<std::vector<int>>(const string& name, const string& key) const {
  if (registry_->hasSpecPar(name))
    return registry_->specPar(name)->value<std::vector<int>>(key);
  else
    return std::vector<int>();
}

template <>
std::vector<std::string> DDFilteredView::get<std::vector<std::string>>(const string& name, const string& key) const {
  if (registry_->hasSpecPar(name))
    return registry_->specPar(name)->value<std::vector<std::string>>(key);
  else
    return std::vector<std::string>();
}

std::vector<double> DDFilteredView::get(const string& name, const string& key) const {
  std::vector<std::string> stringVector = get<std::vector<std::string>>(name, key);
  std::vector<double> doubleVector(stringVector.size());
  std::transform(stringVector.begin(), stringVector.end(), doubleVector.begin(), [](const std::string& val) {
    return std::stod(val);
  });
  return doubleVector;
}

std::string_view DDFilteredView::getString(const std::string& key) const {
  assert(currentFilter_);
  assert(currentFilter_->spec);
  return currentFilter_->spec->strValue(key);
}

DDFilteredView::nav_type DDFilteredView::navPos() const {
  nav_type pos;

  if (not it_.empty()) {
    int level = it_.back().GetLevel();
    for (int i = 1; i <= level; ++i)
      pos.emplace_back(it_.back().GetIndex(i));
  }

  return pos;
}

const std::vector<const Node*> DDFilteredView::geoHistory() const {
  std::vector<const Node*> result;
  if (not it_.empty()) {
    int level = it_.back().GetLevel();
    for (int nit = level; nit > 0; --nit) {
      result.emplace_back(it_.back().GetNode(nit));
    }
  }

  return result;
}

const ExpandedNodes& DDFilteredView::history() {
  assert(registry_);
  nodes_.tags.clear();
  nodes_.offsets.clear();
  nodes_.copyNos.clear();
  bool result(false);

  int level = it_.back().GetLevel();
  for (int nit = level; nit > 0; --nit) {
    for_each(begin(registry_->specpars), end(registry_->specpars), [&](auto const& i) {
      auto k = find_if(begin(i.second.paths), end(i.second.paths), [&](auto const& j) {
        return (isMatch(noNamespace(it_.back().GetNode(nit)->GetVolume()->GetName()),
                        *begin(split(realTopName(j), "/"))) and
                (i.second.hasValue("CopyNoTag") or i.second.hasValue("CopyNoOffset")));
      });
      if (k != end(i.second.paths)) {
        nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
        nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
        nodes_.copyNos.emplace_back(it_.back().GetNode(nit)->GetNumber());
        result = true;
      }
    });
  }

  return nodes_;
}

const TClass* DDFilteredView::getShape() const {
  assert(node_);
  Volume currVol = node_->GetVolume();
  return (currVol->GetShape()->IsA());
}

std::string_view DDFilteredView::name() const { return (volume().volume().name()); }

dd4hep::Solid DDFilteredView::solid() const { return (volume().volume().solid()); }

unsigned short DDFilteredView::copyNum() const { return (volume().copyNumber()); }

std::string_view DDFilteredView::materialName() const { return (volume().material().name()); }
