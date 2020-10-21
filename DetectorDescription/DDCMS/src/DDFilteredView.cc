#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/Detector.h"
#include "DD4hep/Shapes.h"
#include <TGeoBBox.h>
#include <TGeoBoolNode.h>
#include <charconv>

using namespace cms;
using namespace edm;
using namespace std;
using namespace cms::dd;
using namespace dd4hep::dd;

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
      log << "\nSpecPar " << std::string(t.first.data(), t.first.size()) << "\nRegExps { ";
      for (const auto& ki : t.second->paths)
        log << ki << " ";
      log << "};\n ";
      for (const auto& kl : t.second->spars) {
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

void DDFilteredView::mergedSpecifics(DDSpecParRefs const& specPars) {
  currentSpecPar_ = nullptr;
  if (!filters_.empty()) {
    filters_.clear();
    filters_.shrink_to_fit();
  }

  for (const auto& section : specPars) {
    for (const auto& partSelector : section.second->paths) {
      auto const& firstPartName = front(partSelector);
      auto const& filterMatch = find_if(begin(filters_), end(filters_), [&](auto const& it) {
        auto const& key =
            find_if(begin(it->skeys), end(it->skeys), [&](auto const& partName) { return firstPartName == partName; });
        if (key != end(it->skeys)) {
          currentFilter_ = it.get();
          return true;
        }
        return false;
      });
      if (filterMatch == end(filters_)) {
        filters_.emplace_back(make_unique<Filter>());
        filters_.back()->hasNamespace.emplace_back(dd4hep::dd::hasNamespace(firstPartName));
        if (dd4hep::dd::isRegex(firstPartName)) {
          filters_.back()->isRegex.emplace_back(true);
          filters_.back()->index.emplace_back(filters_.back()->keys.size());
          filters_.back()->keys.emplace_back(std::regex(std::begin(firstPartName), std::end(firstPartName)));
        } else {
          filters_.back()->isRegex.emplace_back(false);
          filters_.back()->index.emplace_back(filters_.back()->skeys.size());
        }
        filters_.back()->skeys.emplace_back(firstPartName);
        filters_.back()->up = nullptr;
        filters_.back()->next = nullptr;
        filters_.back()->spec = section.second;
        // initialize current filter if it's empty
        if (currentFilter_ == nullptr) {
          currentFilter_ = filters_.back().get();
        }
      }
      // all next levels
      vector<string_view> toks = split(partSelector, "/");
      for (size_t pos = 1; pos < toks.size(); ++pos) {
        if (currentFilter_->next != nullptr) {
          currentFilter_ = currentFilter_->next.get();
          auto const& key = find_if(begin(currentFilter_->skeys), end(currentFilter_->skeys), [&](auto const& p) {
            return toks.front() == p;
          });
          if (key == end(currentFilter_->skeys)) {
            currentFilter_->hasNamespace.emplace_back(dd4hep::dd::hasNamespace(toks[pos]));
            if (dd4hep::dd::isRegex(toks[pos])) {
              currentFilter_->isRegex.emplace_back(true);
              currentFilter_->index.emplace_back(currentFilter_->keys.size());
              currentFilter_->keys.emplace_back(std::regex(std::begin(toks[pos]), std::end(toks[pos])));
            } else {
              currentFilter_->isRegex.emplace_back(false);
              currentFilter_->index.emplace_back(currentFilter_->skeys.size());
            }
            currentFilter_->skeys.emplace_back(toks[pos]);
          }
        } else {
          auto nextLevelFilter = std::make_unique<Filter>();
          bool isRegex = dd4hep::dd::isRegex(toks[pos]);
          nextLevelFilter->isRegex.emplace_back(isRegex);
          nextLevelFilter->hasNamespace.emplace_back(dd4hep::dd::hasNamespace(toks[pos]));
          if (isRegex) {
            nextLevelFilter->index.emplace_back(filters_.back()->keys.size());
            nextLevelFilter->keys.emplace_back(std::regex(toks[pos].begin(), toks[pos].end()));
          } else {
            nextLevelFilter->index.emplace_back(filters_.back()->skeys.size());
          }
          nextLevelFilter->skeys.emplace_back(toks[pos]);
          nextLevelFilter->next = nullptr;
          nextLevelFilter->up = currentFilter_;
          nextLevelFilter->spec = section.second;

          currentFilter_->next = std::move(nextLevelFilter);
        }
      }
    }
  }
}

void print(const Filter* filter) {
  edm::LogVerbatim("Geometry").log([&](auto& log) {
    for (const auto& it : filter->skeys) {
      log << it << ", ";
    }
  });
}

void DDFilteredView::printFilter() const {
  for (const auto& f : filters_) {
    edm::LogVerbatim("Geometry").log([&](auto& log) {
      log << "\nFilter: ";
      for (const auto& it : f->skeys) {
        log << it << ", ";
      }
      if (f->next) {
        log << "\nNext: ";
        print(&*f->next);
      }
      if (f->up) {
        log << "Up: ";
        print(f->up);
      }
    });
  }
}

bool DDFilteredView::firstChild() {
  currentSpecPar_ = nullptr;

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

bool DDFilteredView::nextChild() {
  currentSpecPar_ = nullptr;

  if (it_.empty()) {
    LogVerbatim("DDFilteredView") << "Iterator vector has zero size.";
    return false;
  }
  it_.back().SetType(0);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (it_.back().GetLevel() <= startLevel_) {
      return false;
    }
    if (accept(noNamespace(node->GetVolume()->GetName()))) {
      node_ = node;
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

std::vector<std::pair<std::string, int>> DDFilteredView::toNodeNames(const std::string& path) {
  std::vector<std::pair<std::string, int>> result;
  std::vector<string_view> names = split(path, "/");
  for (auto it : names) {
    auto name = noNamespace(it);
    int copyNo = -1;
    auto lpos = name.find_first_of('[');
    if (lpos != std::string::npos) {
      auto rpos = name.find_last_of(']');
      if (rpos != std::string::npos) {
        copyNo = nodeCopyNo(name.substr(lpos + 1, rpos - 1));
      }
      name.remove_suffix(name.size() - lpos);
    }
    result.emplace_back(std::string(name.data(), name.size()), copyNo);
  }

  return result;
}

bool DDFilteredView::match(const std::string& path, const std::vector<std::pair<std::string, int>>& names) const {
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
  currentSpecPar_ = nullptr;

  std::vector<std::vector<Node*>> paths;
  if (it_.empty()) {
    LogVerbatim("DDFilteredView") << "Iterator vector has zero size.";
    return paths;
  }
  if (node_ == nullptr) {
    throw cms::Exception("DDFilteredView") << "Can't get children of a null node. Please, call firstChild().";
  }
  it_.back().SetType(0);
  std::vector<std::pair<std::string, int>> names = toNodeNames(selectPath);
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
  currentSpecPar_ = nullptr;

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
    if (dd4hep::dd::accepted(currentFilter_, noNamespace(node->GetVolume()->GetName()))) {
      node_ = node;
      return true;
    }
  } while ((node = it_.back().Next()));

  return false;
}

bool DDFilteredView::nextSibling() {
  currentSpecPar_ = nullptr;

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
      if (dd4hep::dd::accepted(currentFilter_, noNamespace(node->GetVolume()->GetName()))) {
        node_ = node;
        return true;
      }
    } while ((node = it_.back().Next()));

    return false;
  }
}

bool DDFilteredView::sibling() {
  currentSpecPar_ = nullptr;

  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (dd4hep::dd::accepted(currentFilter_, node->GetVolume()->GetName())) {
      node_ = node;
      return true;
    }
  }
  return false;
}

bool DDFilteredView::checkChild() {
  currentSpecPar_ = nullptr;

  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (dd4hep::dd::accepted(currentFilter_, node->GetVolume()->GetName())) {
      return true;
    }
  }
  return false;
}

bool DDFilteredView::parent() {
  currentSpecPar_ = nullptr;
  if (it_.empty() or currentFilter_ == nullptr)
    return false;
  up();
  it_.back().SetType(0);

  return true;
}

bool DDFilteredView::next(int type) {
  currentSpecPar_ = nullptr;

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
  currentSpecPar_ = nullptr;

  if (it_.empty() or currentFilter_ == nullptr)
    return;
  it_.emplace_back(Iterator(it_.back()));
  next(0);
  if (currentFilter_->next)
    currentFilter_ = currentFilter_->next.get();
}

void DDFilteredView::up() {
  currentSpecPar_ = nullptr;

  if (it_.size() > 1 and currentFilter_ != nullptr) {
    it_.pop_back();
    it_.back().SetType(0);
    if (currentFilter_->up)
      currentFilter_ = currentFilter_->up;
  }
}

bool DDFilteredView::accept(std::string_view name) {
  for (const auto& it : filters_) {
    currentFilter_ = it.get();
    if (dd4hep::dd::accepted(currentFilter_, name))
      return true;
  }
  return false;
}

const std::vector<double> DDFilteredView::parameters() const {
  assert(node_);
  Volume currVol = node_->GetVolume();
  // Boolean shapes are a special case
  if (currVol->GetShape()->IsA() == TGeoCompositeShape::Class() and
      not dd4hep::isA<dd4hep::PseudoTrap>(currVol.solid())) {
    const TGeoCompositeShape* shape = static_cast<const TGeoCompositeShape*>(currVol->GetShape());
    const TGeoBoolNode* boolean = shape->GetBoolNode();
    while (boolean->GetLeftShape()->IsA() != TGeoBBox::Class()) {
      boolean = static_cast<const TGeoCompositeShape*>(boolean->GetLeftShape())->GetBoolNode();
    }
    const TGeoBBox* box = static_cast<const TGeoBBox*>(boolean->GetLeftShape());
    return {box->GetDX(), box->GetDY(), box->GetDZ()};
  } else
    return currVol.solid().dimensions();
}

const cms::DDSolidShape DDFilteredView::shape() const {
  assert(node_);
  return cms::dd::value(cms::DDSolidShapeMap, std::string(node_->GetVolume()->GetShape()->GetTitle()));
}

LegacySolidShape DDFilteredView::legacyShape(const cms::DDSolidShape shape) const {
  return cms::dd::value(cms::LegacySolidShapeMap, shape);
}

template <>
std::string_view DDFilteredView::get<string_view>(const string& key) {
  std::string_view result;

  currentSpecPar_ = find(key);
  if (currentSpecPar_ != nullptr) {
    result = currentSpecPar_->strValue(key);
  }
  return result;
}

template <>
double DDFilteredView::get<double>(const string& key) {
  double result(0.0);

  currentSpecPar_ = find(key);
  if (currentSpecPar_ != nullptr) {
    result = getNextValue(key);
  }

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

const int DDFilteredView::level() const {
  int level(0);
  if (not it_.empty()) {
    level = it_.back().GetLevel();
  }
  return level;
}

bool DDFilteredView::goTo(const nav_type& newpos) {
  bool result(false);
  currentSpecPar_ = nullptr;

  // save the current position
  it_.emplace_back(Iterator(it_.back().GetTopVolume()));
  Node* node = nullptr;

  // try to navigate down to the newpos
  for (auto const& i : newpos) {
    it_.back().SetType(0);
    node = it_.back().Next();
    for (int j = 1; j <= i; j++) {
      it_.back().SetType(1);
      node = it_.back().Next();
    }
  }
  if (node != nullptr) {
    node_ = node;
    result = true;
  } else {
    it_.pop_back();
  }

  return result;
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

  int level = it_.back().GetLevel();
  for (int nit = level; nit > 0; --nit) {
    for (auto const& i : registry_->specpars) {
      auto k = find_if(begin(i.second.paths), end(i.second.paths), [&](auto const& j) {
        return (isMatch(noNamespace(it_.back().GetNode(nit)->GetVolume()->GetName()), front(j))) and
               (i.second.hasValue("CopyNoTag") or i.second.hasValue("CopyNoOffset"));
      });
      if (k != end(i.second.paths)) {
        nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
        nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
        nodes_.copyNos.emplace_back(it_.back().GetNode(nit)->GetNumber());
      }
    }
  }

  return nodes_;
}

const DDSpecPar* DDFilteredView::find(const std::string& key) const {
  DDSpecParRefs specParRefs;
  filter(specParRefs, key);

  for (auto const& specPar : specParRefs) {
    auto pos = find_if(begin(specPar.second->paths), end(specPar.second->paths), [&](auto const& partSelector) {
      return matchPath(partSelector);
    });
    if (pos != end(specPar.second->paths)) {
      return specPar.second;
    }
  }

  return nullptr;
}

void DDFilteredView::filter(DDSpecParRefs& refs, const std::string& key) const {
  for (auto const& it : registry_->specpars) {
    if (it.second.hasValue(key) || (it.second.spars.find(key) != end(it.second.spars))) {
      refs.emplace_back(it.first, &it.second);
    }
  }
}

// First name in a path
std::string_view DDFilteredView::front(const std::string_view path) const {
  auto const& lpos = path.find_first_not_of('/');
  if (lpos != path.npos) {
    auto rpos = path.find_first_of('/', lpos);
    if (rpos == path.npos) {
      rpos = path.size();
    }
    return path.substr(lpos, rpos - lpos);
  }

  // throw cms::Exception("Filtered View") << "Path must start with '//'  " << path;
  return path;
}

// Last name in a path
std::string_view DDFilteredView::back(const std::string_view path) const {
  if (auto const& lpos = path.rfind('/') != path.npos) {
    return path.substr(lpos, path.size());
  }

  // throw cms::Exception("Filtered View") << "Path must start with '//'  " << path;
  return path;
}

// Current Iterator level Node name
std::string_view DDFilteredView::nodeNameAt(int level) const {
  assert(!it_.empty());
  assert(it_.back().GetLevel() >= level);
  return it_.back().GetNode(level)->GetVolume()->GetName();
}

// Current Iterator level Node copy number
const int DDFilteredView::nodeCopyNoAt(int level) const {
  assert(!it_.empty());
  assert(it_.back().GetLevel() >= level);
  return it_.back().GetNode(level)->GetNumber();
}

// Compare if name matches a selection pattern that
// may or may not be defined as a regular expression
bool DDFilteredView::compareEqualName(const std::string_view selection, const std::string_view name) const {
  return (!(dd4hep::dd::isRegex(selection)) ? dd4hep::dd::compareEqual(name, selection)
                                            : regex_match(name.begin(), name.end(), regex(std::string(selection))));
}

// Check if both name and it's selection pattern
// contain a namespace and
// remove it if one of them does not
std::tuple<std::string_view, std::string_view> DDFilteredView::alignNamespaces(std::string_view selection,
                                                                               std::string_view name) const {
  auto pos = selection.find(':');
  if (pos == selection.npos) {
    name = noNamespace(name);
  } else {
    if (name.find(':') == name.npos) {
      selection.remove_prefix(pos + 1);
    }
  }
  return std::make_tuple(selection, name);
}

// If a name has an XML-style copy number, e.g. Name[1]
// and compare it to an integer
bool DDFilteredView::compareEqualCopyNumber(const std::string_view name, int copy) const {
  auto pos = name.rfind('[');
  if (pos != name.npos) {
    if (std::stoi(std::string(name.substr(pos + 1, name.rfind(']')))) == copy) {
      return true;
    }
  }

  return false;
}

bool DDFilteredView::matchPath(const std::string_view path) const {
  assert(!it_.empty());
  int level = it_.back().GetLevel();

  auto to = path.size();
  auto from = path.rfind('/');
  bool result = false;
  for (int it = level; from - 1 <= to and it > 0; --it) {
    std::string_view name = nodeNameAt(it);
    std::string_view refname{&path[from + 1], to - from - 1};
    to = from;
    from = path.substr(0, to).rfind('/');

    std::tie(refname, name) = alignNamespaces(refname, name);

    auto pos = refname.rfind('[');
    if (pos != refname.npos) {
      if (!compareEqualCopyNumber(refname, nodeCopyNoAt(it))) {
        result = false;
        break;
      } else {
        refname.remove_suffix(refname.size() - pos);
      }
    }
    if (!compareEqualName(refname, name)) {
      result = false;
      break;
    } else {
      result = true;
    }
  }
  return result;
}

double DDFilteredView::getNextValue(const std::string& key) const {
  double result(0.0);

  if (currentSpecPar_ != nullptr) {
    std::string_view svalue = currentSpecPar_->strValue(key);
    if (!svalue.empty()) {
      result = dd4hep::_toDouble({svalue.data(), svalue.size()});
    } else if (currentSpecPar_->hasValue(key)) {
      auto const& nitem = currentSpecPar_->numpars.find(key);
      if (nitem != end(currentSpecPar_->numpars)) {
        result = nitem->second[0];
      }
    }
  }

  return result;
}

std::string_view DDFilteredView::name() const {
  return (node_ == nullptr ? std::string_view() : (volume().volume().name()));
}

dd4hep::Solid DDFilteredView::solid() const { return (volume().volume().solid()); }

unsigned short DDFilteredView::copyNum() const { return (volume().copyNumber()); }

std::string_view DDFilteredView::materialName() const { return (volume().material().name()); }

std::ostream& operator<<(std::ostream& os, const std::vector<const cms::Node*>& hst) {
  for (auto nd = hst.rbegin(); nd != hst.rend(); ++nd)
    os << "/" << (*nd)->GetName();
  return os;
}
