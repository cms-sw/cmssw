#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/Detector.h"
#include "DD4hep/Shapes.h"
#include <TGeoBBox.h>
#include <TGeoBoolNode.h>
#include <vector>

using namespace cms;
using namespace edm;
using namespace std;
using namespace cms::dd;

dd4hep::Solid DDSolid::solidA() const {
  if (dd4hep::isA<dd4hep::SubtractionSolid>(solid_) || dd4hep::isA<dd4hep::UnionSolid>(solid_) ||
      dd4hep::isA<dd4hep::IntersectionSolid>(solid_)) {
    const TGeoCompositeShape* sh = (const TGeoCompositeShape*)solid_.ptr();
    const TGeoBoolNode* boolean = sh->GetBoolNode();
    TGeoShape* solidA = boolean->GetLeftShape();
    return dd4hep::Solid(solidA);
  }
  return solid_;
}

dd4hep::Solid DDSolid::solidB() const {
  if (dd4hep::isA<dd4hep::SubtractionSolid>(solid_) || dd4hep::isA<dd4hep::UnionSolid>(solid_) ||
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

DDFilteredView::DDFilteredView(const DDCompactView& cpv, const DDFilter& attribute) : registry_(&cpv.specpars()) {
  it_.emplace_back(Iterator(cpv.detector()->worldVolume()));
  registry_->filter(refs_, attribute);
  mergedSpecifics(refs_);
  LogVerbatim("Geometry").log([&](auto& log) {
    log << "Filtered DD SpecPar Registry size: " << refs_.size() << "\n";
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

  for (int i = it_.back().GetLevel(); i > 0; --i) {
    result.emplace_back(it_.back().GetNode(i)->GetNumber());
  }

  return result;
}

const Double_t* DDFilteredView::trans() const { return it_.back().GetCurrentMatrix()->GetTranslation(); }

const Translation DDFilteredView::translation() const {
  const Double_t* translation = it_.back().GetCurrentMatrix()->GetTranslation();
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

  LogVerbatim("DDFilteredView") << "Rotation matrix components (1st 3) = " << rotation[0] << ", " << rotation[1] << ", "
                                << rotation[2];
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
  LogVerbatim("DDFilteredView") << "Rotation matrix components (1st 3) = " << rotation[0] << ", " << rotation[1] << ", "
                                << rotation[2];
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
  for (const auto& i : specs) {
    for (const auto& j : i->paths) {
      vector<string_view> toks = split(j, "/");
      auto const& filter = find_if(begin(filters_), end(filters_), [&](auto const& f) {
        auto const& k = find(begin(f->keys), end(f->keys), toks.front());
        if (k != end(f->keys)) {
          currentFilter_ = f.get();
          return true;
        }
        return false;
      });
      if (filter == end(filters_)) {
        filters_.emplace_back(unique_ptr<Filter>(new Filter{{toks.front()}, nullptr, nullptr, i}));
        currentFilter_ = filters_.back().get();
      }
      // all next levels
      for (size_t pos = 1; pos < toks.size(); ++pos) {
        if (currentFilter_->next != nullptr) {
          currentFilter_ = currentFilter_->next.get();
          auto const& l = find(begin(currentFilter_->keys), end(currentFilter_->keys), toks[pos]);
          if (l == end(currentFilter_->keys)) {
            currentFilter_->keys.emplace_back(toks[pos]);
          }
        } else {
          currentFilter_->next.reset(new Filter{{toks[pos]}, nullptr, currentFilter_, i});
        }
      }
    }
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
      return true;
    }
  }
  LogVerbatim("DDFilteredView") << "Search for first child failed.";
  return false;
}

bool DDFilteredView::firstSibling() {
  assert(node_);
  if (it_.empty() || currentFilter_ == nullptr)
    return false;
  Node* node = nullptr;
  if (next(0) == false)
    return false;
  node = node_;
  it_.emplace_back(Iterator(it_.back()));
  it_.back().SetType(1);
  if (currentFilter_ != nullptr && currentFilter_->next != nullptr)
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
  if (it_.empty() || currentFilter_ == nullptr)
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
  if (it_.empty() || currentFilter_ == nullptr)
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
  if (it_.empty() || currentFilter_ == nullptr)
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
  if (it_.empty() || currentFilter_ == nullptr)
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
  if (it_.empty() || currentFilter_ == nullptr)
    return;
  it_.emplace_back(Iterator(it_.back()));
  next(0);
  if (currentFilter_->next)
    currentFilter_ = currentFilter_->next.get();
}

void DDFilteredView::up() {
  if (it_.size() > 1 && currentFilter_ != nullptr) {
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
  Volume currVol = node_->GetVolume();
  // Boolean shapes are a special case
  if (currVol->GetShape()->IsA() == TGeoCompositeShape::Class()) {
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
  return cms::dd::value(cms::DDSolidShapeMap, std::string(node_->GetVolume()->GetShape()->GetTitle()));
}

LegacySolidShape DDFilteredView::legacyShape(const cms::DDSolidShape shape) const {
  return cms::dd::value(cms::LegacySolidShapeMap, shape);
}

template <>
std::string_view DDFilteredView::get<string_view>(const string& key) const {
  std::string_view result;
  DDSpecParRefs refs;
  registry_->filter(refs, key);
  int level = it_.back().GetLevel();
  for_each(begin(refs), end(refs), [&](auto const& i) {
    auto k = find_if(begin(i->paths), end(i->paths), [&](auto const& j) {
      auto const& names = split(realTopName(j), "/");
      int count = names.size();
      bool flag = false;
      for (int nit = level; count > 0 && nit > 0; --nit) {
        if (!compareEqual(noNamespace(it_.back().GetNode(nit)->GetVolume()->GetName()), names[--count])) {
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
std::vector<std::string> DDFilteredView::get<std::vector<std::string>>(const string& name, const string& key) const {
  if (registry_->hasSpecPar(name))
    return registry_->specPar(name)->value<std::vector<std::string>>(key);
  else
    return std::vector<std::string>();
}

std::string_view DDFilteredView::getString(const std::string& key) const {
  assert(currentFilter_);
  assert(currentFilter_->spec);
  return currentFilter_->spec->strValue(key);
}

DDFilteredView::nav_type DDFilteredView::navPos() const {
  Int_t level = it_.back().GetLevel();
  nav_type pos(level);
  for (Int_t i = 1; i <= level; ++i)
    pos[i] = it_.back().GetIndex(i);

  return pos;
}

bool DDFilteredView::addPath(Node* const node) {
  assert(registry_);
  node_ = node;
  nodes_.tags.clear();
  nodes_.offsets.clear();
  nodes_.copyNos.clear();
  bool result(false);

  int level = it_.back().GetLevel();
  for (int nit = level; nit > 0; --nit) {
    for_each(begin(registry_->specpars), end(registry_->specpars), [&](auto const& i) {
      auto k = find_if(begin(i.second.paths), end(i.second.paths), [&](auto const& j) {
        return (compareEqual(noNamespace(it_.back().GetNode(nit)->GetVolume()->GetName()),
                             *begin(split(realTopName(j), "/"))) &&
                (i.second.hasValue("CopyNoTag") || i.second.hasValue("CopyNoOffset")));
      });
      if (k != end(i.second.paths)) {
        nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
        nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
        nodes_.copyNos.emplace_back(it_.back().GetNode(nit)->GetNumber());
        result = true;
      }
    });
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
        return (compareEqual(noNamespace(it_.back().GetNode(nit)->GetVolume()->GetName()),
                             *begin(split(realTopName(j), "/"))) &&
                (i.second.hasValue("CopyNoTag") || i.second.hasValue("CopyNoOffset")));
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

bool DDFilteredView::addNode(Node* const node) {
  assert(registry_);
  node_ = node;
  bool result(false);
  for_each(begin(registry_->specpars), end(registry_->specpars), [&](auto const& i) {
    auto k = find_if(begin(i.second.paths), end(i.second.paths), [&](auto const& j) {
      return (compareEqual(node_->GetVolume()->GetName(), *begin(split(realTopName(j), "/"))) &&
              (i.second.hasValue("CopyNoTag") || i.second.hasValue("CopyNoOffset")));
    });
    if (k != end(i.second.paths)) {
      nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
      nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
      nodes_.copyNos.emplace_back(node_->GetNumber());
      result = true;
    }
  });
  return result;
}

const TClass* DDFilteredView::getShape() const {
  assert(node_);
  Volume currVol = node_->GetVolume();
  return (currVol->GetShape()->IsA());
}

bool DDFilteredView::isABox() const { return isA<dd4hep::Box>(); }

bool DDFilteredView::isAConeSeg() const { return isA<dd4hep::ConeSegment>(); }

bool DDFilteredView::isAPseudoTrap() const { return isA<dd4hep::PseudoTrap>(); }

bool DDFilteredView::isATrapezoid() const { return isA<dd4hep::Trap>(); }

bool DDFilteredView::isATruncTube() const { return isA<dd4hep::TruncatedTube>(); }

bool DDFilteredView::isATubeSeg() const { return isA<dd4hep::Tube>(); }

bool DDFilteredView::isASubtraction() const {
  return (isA<dd4hep::SubtractionSolid>() && !isA<dd4hep::TruncatedTube>() && !isA<dd4hep::PseudoTrap>());
}

std::string_view DDFilteredView::name() const { return (volume().volume().name()); }

dd4hep::Solid DDFilteredView::solid() const { return (volume().volume().solid()); }

unsigned short DDFilteredView::copyNum() const { return (volume().copyNumber()); }

std::string_view DDFilteredView::materialName() const { return (volume().material().name()); }
