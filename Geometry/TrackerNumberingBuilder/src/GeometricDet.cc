#include <DD4hep/DD4hepUnits.h>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Rounding.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/TrackerShapeToBounds.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"

#include <cfloat>
#include <string>

namespace {

  const std::string strue("true");

  template <typename DDView>
  double getDouble(const char* s, DDView const& ev) {
    DDValue val(s);
    std::vector<const DDsvalues_type*> result;
    ev.specificsV(result);
    std::vector<const DDsvalues_type*>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it) {
      foundIt = DDfetch(*it, val);
      if (foundIt)
        break;
    }
    if (foundIt) {
      const std::vector<std::string>& temp = val.strings();
      if (temp.size() != 1) {
        throw cms::Exception("Configuration") << "I need 1 " << s << " tags";
      }
      return double(::atof(temp[0].c_str()));
    }
    return 0;
  }

  template <typename DDView>
  std::string getString(const char* s, DDView const& ev) {
    DDValue val(s);
    std::vector<const DDsvalues_type*> result;
    ev.specificsV(result);
    std::vector<const DDsvalues_type*>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it) {
      foundIt = DDfetch(*it, val);
      if (foundIt)
        break;
    }
    if (foundIt) {
      const std::vector<std::string>& temp = val.strings();
      if (temp.size() != 1) {
        throw cms::Exception("Configuration") << "I need 1 " << s << " tags";
      }
      return temp[0];
    }
    return "NotFound";
  }
}  // namespace

/**
 * What to do in the destructor?
 * destroy all the daughters!
 */
GeometricDet::~GeometricDet() { deleteComponents(); }

/*
  Constructor from old DD Filtered view.
*/
GeometricDet::GeometricDet(DDFilteredView* fv, GeometricEnumType type)
    : ddname_(fv->name()),
      type_(type),
      ddd_(),  // To remove after DetExtra is removed (not used)
      trans_(fv->translation()),
      rho_(trans_.Rho()),
      phi_(trans_.Phi()),
      rot_(fv->rotation()),
      shape_(cms::dd::name_from_value(cms::LegacySolidShapeMap, fv->shape())),
      params_(fv->parameters()),
      isFromDD4hep_(false) {
  //  workaround instead of this at initialization
  const DDFilteredView::nav_type& nt = fv->navPos();
  ddd_ = nav_type(nt.begin(), nt.end());

  // Only look for sensor-related info on sensor volumes!
  if (type_ == DetUnit) {
    radLength_ = getDouble("TrackerRadLength", *fv);
    xi_ = getDouble("TrackerXi", *fv);
    pixROCRows_ = getDouble("PixelROCRows", *fv);
    pixROCCols_ = getDouble("PixelROCCols", *fv);
    pixROCx_ = getDouble("PixelROC_X", *fv);
    pixROCy_ = getDouble("PixelROC_Y", *fv);
    stereo_ = (getString("TrackerStereoDetectors", *fv) == strue);
    isLowerSensor_ = (getString("TrackerLowerDetectors", *fv) == strue);
    isUpperSensor_ = (getString("TrackerUpperDetectors", *fv) == strue);
    siliconAPVNum_ = getDouble("SiliconAPVNumber", *fv);
  }
}

/*
  Constructor from DD4hep Filtered view.
*/
GeometricDet::GeometricDet(cms::DDFilteredView* fv, GeometricEnumType type)
    : ddname_(dd4hep::dd::noNamespace(fv->name())),
      type_(type),
      ddd_(fv->navPos()),  // To remove after DetExtra is removed (not used)
      trans_((fv->translation()) / dd4hep::mm),
      rho_(trans_.Rho()),
      phi_(trans_.Phi()),
      rot_(fv->rotation()),
      shape_(fv->shape()),
      params_(computeLegacyShapeParameters(shape_, fv->solid())),
      isFromDD4hep_(true) {
  using namespace angle_units::operators;
  if (almostEqual(phi_, -1._pi, 10)) {
    phi_ = 1._pi;
    // Standardize phi values of |pi| to be always +pi instead of sometimes -pi.
  }

  // Only look for sensor-related info on sensor volumes!
  if (type_ == DetUnit) {
    // IT sensors only (NB: hence could add a branch here, but not a critical part on perf)
    pixROCRows_ = fv->get<double>("PixelROCRows");
    pixROCCols_ = fv->get<double>("PixelROCCols");
    pixROCx_ = fv->get<double>("PixelROC_X");
    pixROCy_ = fv->get<double>("PixelROC_Y");

    // Phase 1 OT sensors only (NB: hence could add a branch here, but not a critical part on perf)
    stereo_ = (fv->get<std::string_view>("TrackerStereoDetectors") == strue);
    siliconAPVNum_ = fv->get<double>("SiliconAPVNumber");

    // Phase 2 OT sensors only (NB: hence could add a branch here, but not a critical part on perf)
    isLowerSensor_ = (fv->get<std::string_view>("TrackerLowerDetectors") == strue);
    isUpperSensor_ = (fv->get<std::string_view>("TrackerUpperDetectors") == strue);

    // All sensors: IT or OT, Phase 1 or Phase 2 (NB: critical part on perf)
    fv->findSpecPar("TrackerRadLength", "TrackerXi");
    radLength_ = fv->getNextValue("TrackerRadLength");
    xi_ = fv->getNextValue("TrackerXi");
  }
}

/*
  Constructor from persistent version (DB).
*/
GeometricDet::GeometricDet(const PGeometricDet::Item& onePGD, GeometricEnumType type)
    : ddname_(onePGD._name),
      type_(type),
      ddd_(),
      geographicalID_(onePGD._geographicalID),
      trans_(onePGD._x, onePGD._y, onePGD._z),
      rho_(onePGD._rho),
      phi_(onePGD._phi),
      rot_(onePGD._a11,
           onePGD._a12,
           onePGD._a13,
           onePGD._a21,
           onePGD._a22,
           onePGD._a23,
           onePGD._a31,
           onePGD._a32,
           onePGD._a33),
      shape_(cms::dd::name_from_value(cms::LegacySolidShapeMap, static_cast<LegacySolidShape>(onePGD._shape))),
      params_(),
      radLength_(onePGD._radLength),
      xi_(onePGD._xi),
      pixROCRows_(onePGD._pixROCRows),
      pixROCCols_(onePGD._pixROCCols),
      pixROCx_(onePGD._pixROCx),
      pixROCy_(onePGD._pixROCy),
      stereo_(onePGD._stereo),
      siliconAPVNum_(onePGD._siliconAPVNum)
// NB: what about new data members isLowerSensor_, isUpperSensor_, isFromDD4hep_?
// They are presently not added to PGeometricDet (no change in info stored into DB).
{
  // Solid shape parameters: only for box (1) and trapezoid (3)
  if (onePGD._shape == 1 || onePGD._shape == 3) {
    params_.reserve(11);
    params_.emplace_back(onePGD._params0);
    params_.emplace_back(onePGD._params1);
    params_.emplace_back(onePGD._params2);
    params_.emplace_back(onePGD._params3);
    params_.emplace_back(onePGD._params4);
    params_.emplace_back(onePGD._params5);
    params_.emplace_back(onePGD._params6);
    params_.emplace_back(onePGD._params7);
    params_.emplace_back(onePGD._params8);
    params_.emplace_back(onePGD._params9);
    params_.emplace_back(onePGD._params10);
  }

  ddd_.reserve(onePGD._numnt);
  ddd_.emplace_back(onePGD._nt0);
  ddd_.emplace_back(onePGD._nt1);
  ddd_.emplace_back(onePGD._nt2);
  ddd_.emplace_back(onePGD._nt3);
  if (onePGD._numnt > 4) {
    ddd_.emplace_back(onePGD._nt4);
    if (onePGD._numnt > 5) {
      ddd_.emplace_back(onePGD._nt5);
      if (onePGD._numnt > 6) {
        ddd_.emplace_back(onePGD._nt6);
        if (onePGD._numnt > 7) {
          ddd_.emplace_back(onePGD._nt7);
          if (onePGD._numnt > 8) {
            ddd_.emplace_back(onePGD._nt8);
            if (onePGD._numnt > 9) {
              ddd_.emplace_back(onePGD._nt9);
              if (onePGD._numnt > 10) {
                ddd_.emplace_back(onePGD._nt10);
              }
            }
          }
        }
      }
    }
  }
}

std::unique_ptr<Bounds> GeometricDet::bounds() const {
  TrackerShapeToBounds shapeToBounds;
  return std::unique_ptr<Bounds>(shapeToBounds.buildBounds(shape_, params_));
}

GeometricDet::Position GeometricDet::positionBounds() const {
  Position pos_(static_cast<float>(geant_units::operators::convertMmToCm(trans_.x())),
                static_cast<float>(geant_units::operators::convertMmToCm(trans_.y())),
                static_cast<float>(geant_units::operators::convertMmToCm(trans_.z())));
  return pos_;
}

GeometricDet::Rotation GeometricDet::rotationBounds() const {
  Translation x, y, z;
  rot_.GetComponents(x, y, z);
  Rotation rotation_(float(x.X()),
                     float(x.Y()),
                     float(x.Z()),
                     float(y.X()),
                     float(y.Y()),
                     float(y.Z()),
                     float(z.X()),
                     float(z.Y()),
                     float(z.Z()));
  return rotation_;
}

GeometricDet::ConstGeometricDetContainer GeometricDet::deepComponents() const {
  // iterate on all the DESCENDANTS!!
  ConstGeometricDetContainer temp_;
  deepComponents(temp_);
  return temp_;
}

void GeometricDet::deepComponents(ConstGeometricDetContainer& cont) const {
  if (isLeaf())
    cont.emplace_back(this);
  else
    std::for_each(container_.begin(), container_.end(), [&](const GeometricDet* iDet) { iDet->deepComponents(cont); });
}

void GeometricDet::addComponents(GeometricDetContainer const& cont) {
  container_.reserve(container_.size() + cont.size());
  std::copy(cont.begin(), cont.end(), back_inserter(container_));
}

void GeometricDet::addComponents(ConstGeometricDetContainer const& cont) {
  container_.reserve(container_.size() + cont.size());
  std::copy(cont.begin(), cont.end(), back_inserter(container_));
}

void GeometricDet::addComponent(GeometricDet* det) { container_.emplace_back(det); }

namespace {
  struct Deleter {
    void operator()(GeometricDet const* det) const { delete const_cast<GeometricDet*>(det); }
  };
}  // namespace

void GeometricDet::deleteComponents() {
  std::for_each(container_.begin(), container_.end(), Deleter());
  container_.clear();
}

/*
 * PRIVATE
 */

/*
 * DD4hep.
 * Keep order and units of parameters same as old DD, to avoid numerous regressions.
 * Shape parameters of interest, and those to be stored in DB, are only from boxes, trapezoids, and tubs.
 * Hence, they are the only shapes treated here. 
 * params() will complain, if the parameters of any other shape are accessed.
 */
std::vector<double> GeometricDet::computeLegacyShapeParameters(const cms::DDSolidShape& mySolidShape,
                                                               const dd4hep::Solid& mySolid) const {
  std::vector<double> myOldDDShapeParameters;

  // Box
  if (mySolidShape == cms::DDSolidShape::ddbox) {
    const dd4hep::Box& myBox = dd4hep::Box(mySolid);
    myOldDDShapeParameters = {(myBox.x()) / dd4hep::mm, (myBox.y()) / dd4hep::mm, (myBox.z()) / dd4hep::mm};
  }

  // Trapezoid
  else if (mySolidShape == cms::DDSolidShape::ddtrap) {
    const dd4hep::Trap& myTrap = dd4hep::Trap(mySolid);
    myOldDDShapeParameters = {(myTrap->GetDZ()) / dd4hep::mm,
                              static_cast<double>(angle_units::operators::convertDegToRad(myTrap->GetTheta())),
                              static_cast<double>(angle_units::operators::convertDegToRad(myTrap->GetPhi())),
                              (myTrap->GetH1()) / dd4hep::mm,
                              (myTrap->GetBl1()) / dd4hep::mm,
                              (myTrap->GetTl1()) / dd4hep::mm,
                              static_cast<double>(angle_units::operators::convertDegToRad(myTrap->GetAlpha1())),
                              (myTrap->GetH2()) / dd4hep::mm,
                              (myTrap->GetBl2()) / dd4hep::mm,
                              (myTrap->GetTl2()) / dd4hep::mm,
                              static_cast<double>(angle_units::operators::convertDegToRad(myTrap->GetAlpha2()))};
  }

  // Tub
  else if (mySolidShape == cms::DDSolidShape::ddtubs) {
    const dd4hep::Tube& myTube = dd4hep::Tube(mySolid);
    double phi1 = static_cast<double>(angle_units::operators::convertDegToRad(myTube->GetPhi1()));
    phi1 = cms_rounding::roundIfNear0(phi1);
    if (phi1 != 0.0) {
      using namespace angle0to2pi;
      phi1 = make0To2pi(phi1) - 2._pi;
      // Turn positive angle to negative
    }
    myOldDDShapeParameters = {
        (myTube->GetDz()) / dd4hep::mm,
        (myTube->GetRmin()) / dd4hep::mm,
        (myTube->GetRmax()) / dd4hep::mm,
        phi1,
        static_cast<double>(angle_units::operators::convertDegToRad(myTube->GetPhi2() - myTube->GetPhi1()))};
  }

  return myOldDDShapeParameters;
}

std::string GeometricDet::printNavType(int const* n, size_t sz) {
  std::ostringstream oss;
  oss << '(';
  for (int const* it = n; it != n + sz; ++it) {
    oss << *it << ',';
  }
  oss << ')';
  return oss.str();
}
