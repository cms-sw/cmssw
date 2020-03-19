// -*- C++ -*-
//
// Package:     Geometry
// Class  :     TGeoMgrFromDdd
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:
//         Created:  Fri Jul  2 16:11:42 CEST 2010
//

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoCompositeShape.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoCone.h"
#include "TGeoBoolNode.h"
#include "TGeoTube.h"
#include "TGeoArb8.h"
#include "TGeoTrd2.h"
#include "TGeoTorus.h"
#include "TGeoEltu.h"
#include "TGeoXtru.h"

#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationZ.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <cmath>

class TGeoMgrFromDdd : public edm::ESProducer {
public:
  TGeoMgrFromDdd(const edm::ParameterSet&);
  TGeoMgrFromDdd(const TGeoMgrFromDdd&) = delete;
  const TGeoMgrFromDdd& operator=(const TGeoMgrFromDdd&) = delete;

  using ReturnType = std::unique_ptr<TGeoManager>;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  ReturnType produce(const DisplayGeomRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  TGeoManager* createManager(int level);

  TGeoShape* createShape(const std::string& iName, const DDSolid& iSolid);
  TGeoVolume* createVolume(const std::string& iName, const DDSolid& iSolid, const DDMaterial& iMaterial);
  TGeoMaterial* createMaterial(const DDMaterial& iMaterial);

  // ---------- member data --------------------------------

  const int m_level;
  const bool m_verbose;
  const bool m_fullname;
  std::string m_TGeoName;
  std::string m_TGeoTitle;

  std::map<std::string, TGeoShape*> nameToShape_;
  std::map<std::string, TGeoVolume*> nameToVolume_;
  std::map<std::string, TGeoMaterial*> nameToMaterial_;
  std::map<std::string, TGeoMedium*> nameToMedium_;

  edm::ESGetToken<DDCompactView, IdealGeometryRecord> viewToken_;
};

TGeoMgrFromDdd::TGeoMgrFromDdd(const edm::ParameterSet& pset)
    : m_level(pset.getUntrackedParameter<int>("level")),
      m_verbose(pset.getUntrackedParameter<bool>("verbose")),
      m_fullname(pset.getUntrackedParameter<bool>("fullName")) {
  // The following line is needed to tell the framework what data is
  // being produced.
  setWhatProduced(this).setConsumes(viewToken_);
}

void TGeoMgrFromDdd::fillDescriptions(edm::ConfigurationDescriptions& conf) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<int>("level", 10)->setComment("How deep into the geometry hierarchy should the conversion go.");
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<bool>("fullName", true)->setComment("use fillname() instead of name() when generating node names");

  conf.add("TGeoMgrFromDdd", desc);
}

//==============================================================================
// Local helpers
//==============================================================================

namespace {
  TGeoCombiTrans* createPlacement(const DDRotationMatrix& iRot, const DDTranslation& iTrans) {
    double elements[9];
    iRot.GetComponents(elements);
    TGeoRotation r;
    r.SetMatrix(elements);

    TGeoTranslation t(iTrans.x() / cm, iTrans.y() / cm, iTrans.z() / cm);

    return new TGeoCombiTrans(t, r);
  }
}  // namespace

//==============================================================================
// public member functions
//==============================================================================

TGeoMgrFromDdd::ReturnType TGeoMgrFromDdd::produce(const DisplayGeomRecord& iRecord) {
  using namespace edm;

  ESTransientHandle<DDCompactView> viewH = iRecord.getTransientHandle(viewToken_);

  if (!viewH.isValid()) {
    return std::unique_ptr<TGeoManager>();
  }

  TGeoManager* geo_mgr = new TGeoManager("cmsGeo", "CMS Detector");
  // NOTE: the default constructor does not create the identity matrix
  if (gGeoIdentity == nullptr) {
    gGeoIdentity = new TGeoIdentity("Identity");
  }

  std::cout << "about to initialize the DDCompactView walker"
            << " with a root node " << viewH->root() << std::endl;

  auto walker = viewH->walker();
  auto info = walker.current();

  // The top most item is actually the volume holding both the
  // geometry AND the magnetic field volumes!
  walker.firstChild();
  if (!walker.firstChild()) {
    return std::unique_ptr<TGeoManager>();
  }

  TGeoVolume* top = (m_fullname ? createVolume(info.first.name().fullname(), info.first.solid(), info.first.material())
                                : createVolume(info.first.name().name(), info.first.solid(), info.first.material()));

  if (top == nullptr) {
    return std::unique_ptr<TGeoManager>();
  }

  geo_mgr->SetTopVolume(top);
  // ROOT chokes unless colors are assigned
  top->SetVisibility(kFALSE);
  top->SetLineColor(kBlue);

  std::vector<TGeoVolume*> parentStack;
  parentStack.push_back(top);

  do {
    auto info = walker.current();

    if (m_verbose) {
      for (unsigned int i = 0; i < parentStack.size(); ++i) {
        std::cout << " ";
      }
      std::cout << info.first.name() << " " << info.second->copyno() << " "
                << DDSolidShapesName::name(info.first.solid().shape()) << std::endl;
    }

    std::string name = m_fullname ? info.first.name().fullname() : info.first.name().name();
    bool childAlreadyExists = (nullptr != nameToVolume_[name]);
    TGeoVolume* child = createVolume(name, info.first.solid(), info.first.material());
    if (nullptr != child && info.second != nullptr) {
      parentStack.back()->AddNode(
          child, info.second->copyno(), createPlacement(info.second->rotation(), info.second->translation()));
      child->SetLineColor(kBlue);
    } else {
      if (info.second == nullptr) {
        break;
      }
    }
    if (nullptr == child || childAlreadyExists || m_level == int(parentStack.size())) {
      if (nullptr != child) {
        child->SetLineColor(kRed);
      }
      //stop descending
      if (!walker.nextSibling()) {
        while (walker.parent()) {
          parentStack.pop_back();
          if (walker.nextSibling()) {
            break;
          }
        }
      }
    } else {
      if (walker.firstChild()) {
        parentStack.push_back(child);
      } else {
        if (!walker.nextSibling()) {
          while (walker.parent()) {
            parentStack.pop_back();
            if (walker.nextSibling()) {
              break;
            }
          }
        }
      }
    }
  } while (!parentStack.empty());

  geo_mgr->CloseGeometry();

  geo_mgr->DefaultColors();

  nameToShape_.clear();
  nameToVolume_.clear();
  nameToMaterial_.clear();
  nameToMedium_.clear();

  return std::unique_ptr<TGeoManager>(geo_mgr);
}

//==============================================================================
// private member functions
//==============================================================================

TGeoShape* TGeoMgrFromDdd::createShape(const std::string& iName, const DDSolid& iSolid) {
  LogDebug("TGeoMgrFromDdd::createShape") << "with name: " << iName << " and solid: " << iSolid;

  DDBase<DDName, DDI::Solid*>::def_type defined(iSolid.isDefined());
  if (!defined.first)
    throw cms::Exception("TGeoMgrFromDdd::createShape * solid " + iName + " is not declared * ");
  if (!defined.second)
    throw cms::Exception("TGeoMgrFromDdd::createShape * solid " + defined.first->name() + " is not defined *");

  TGeoShape* rSolid = nameToShape_[iName];
  if (rSolid == nullptr) {
    const std::vector<double>& params = iSolid.parameters();
    switch (iSolid.shape()) {
      case DDSolidShape::ddbox:
        rSolid = new TGeoBBox(iName.c_str(), params[0] / cm, params[1] / cm, params[2] / cm);
        break;
      case DDSolidShape::ddcons:
        rSolid = new TGeoConeSeg(iName.c_str(),
                                 params[0] / cm,
                                 params[1] / cm,
                                 params[2] / cm,
                                 params[3] / cm,
                                 params[4] / cm,
                                 params[5] / deg,
                                 params[6] / deg + params[5] / deg);
        break;
      case DDSolidShape::ddtubs:
        //Order in params is  zhalf,rIn,rOut,startPhi,deltaPhi
        rSolid = new TGeoTubeSeg(iName.c_str(),
                                 params[1] / cm,
                                 params[2] / cm,
                                 params[0] / cm,
                                 params[3] / deg,
                                 params[3] / deg + params[4] / deg);
        break;
      case DDSolidShape::ddcuttubs:
        //Order in params is  zhalf,rIn,rOut,startPhi,deltaPhi,lx,ly,lz,tx,ty,tz
        rSolid = new TGeoCtub(iName.c_str(),
                              params[1] / cm,
                              params[2] / cm,
                              params[0] / cm,
                              params[3] / deg,
                              params[3] / deg + params[4] / deg,
                              params[5],
                              params[6],
                              params[7],
                              params[8],
                              params[9],
                              params[10]);
        break;
      case DDSolidShape::ddtrap:
        rSolid = new TGeoTrap(iName.c_str(),
                              params[0] / cm,     //dz
                              params[1] / deg,    //theta
                              params[2] / deg,    //phi
                              params[3] / cm,     //dy1
                              params[4] / cm,     //dx1
                              params[5] / cm,     //dx2
                              params[6] / deg,    //alpha1
                              params[7] / cm,     //dy2
                              params[8] / cm,     //dx3
                              params[9] / cm,     //dx4
                              params[10] / deg);  //alpha2
        break;
      case DDSolidShape::ddpolycone_rrz:
        rSolid = new TGeoPcon(iName.c_str(), params[0] / deg, params[1] / deg, (params.size() - 2) / 3);
        {
          std::vector<double> temp(params.size() + 1);
          temp.reserve(params.size() + 1);
          temp[0] = params[0] / deg;
          temp[1] = params[1] / deg;
          temp[2] = (params.size() - 2) / 3;
          std::copy(params.begin() + 2, params.end(), temp.begin() + 3);
          for (std::vector<double>::iterator it = temp.begin() + 3; it != temp.end(); ++it) {
            *it /= cm;
          }
          rSolid->SetDimensions(&(*(temp.begin())));
        }
        break;
      case DDSolidShape::ddpolyhedra_rrz:
        rSolid = new TGeoPgon(
            iName.c_str(), params[1] / deg, params[2] / deg, static_cast<int>(params[0]), (params.size() - 3) / 3);
        {
          std::vector<double> temp(params.size() + 1);
          temp[0] = params[1] / deg;
          temp[1] = params[2] / deg;
          temp[2] = params[0];
          temp[3] = (params.size() - 3) / 3;
          std::copy(params.begin() + 3, params.end(), temp.begin() + 4);
          for (std::vector<double>::iterator it = temp.begin() + 4; it != temp.end(); ++it) {
            *it /= cm;
          }
          rSolid->SetDimensions(&(*(temp.begin())));
        }
        break;
      case DDSolidShape::ddextrudedpolygon: {
        DDExtrudedPolygon extrPgon(iSolid);
        std::vector<double> x = extrPgon.xVec();
        std::transform(x.begin(), x.end(), x.begin(), [](double d) { return d / cm; });
        std::vector<double> y = extrPgon.yVec();
        std::transform(y.begin(), y.end(), y.begin(), [](double d) { return d / cm; });
        std::vector<double> z = extrPgon.zVec();
        std::vector<double> zx = extrPgon.zxVec();
        std::vector<double> zy = extrPgon.zyVec();
        std::vector<double> zscale = extrPgon.zscaleVec();

        TGeoXtru* mySolid = new TGeoXtru(z.size());
        mySolid->DefinePolygon(x.size(), &(*x.begin()), &(*y.begin()));
        for (size_t i = 0; i < params[0]; ++i) {
          mySolid->DefineSection(i, z[i] / cm, zx[i] / cm, zy[i] / cm, zscale[i]);
        }

        rSolid = mySolid;
      } break;
      case DDSolidShape::ddpseudotrap: {
        //implementation taken from SimG4Core/Geometry/src/DDG4SolidConverter.cc
        const static DDRotationMatrix s_rot(ROOT::Math::RotationX(90. * deg));
        DDPseudoTrap pt(iSolid);

        double r = pt.radius();
        bool atMinusZ = pt.atMinusZ();
        double x = 0;
        double h = 0;
        bool intersec = false;  // union or intersection solid

        if (atMinusZ) {
          x = pt.x1();  // tubs radius
        } else {
          x = pt.x2();  // tubs radius
        }
        double halfOpeningAngle = asin(x / std::abs(r)) / deg;
        double displacement = 0;
        double startPhi = 0;
        /* calculate the displacement of the tubs w.r.t. to the trap,
	       determine the opening angle of the tubs */
        double delta = sqrt(r * r - x * x);
        std::string name = m_fullname ? pt.name().fullname() : pt.name().name();

        if (r < 0 && std::abs(r) >= x) {
          intersec = true;                            // intersection solid
          h = pt.y1() < pt.y2() ? pt.y2() : pt.y1();  // tubs half height
          h += h / 20.;                               // enlarge a bit - for subtraction solid
          if (atMinusZ) {
            displacement = -pt.halfZ() - delta;
            startPhi = 90. - halfOpeningAngle;
          } else {
            displacement = pt.halfZ() + delta;
            startPhi = -90. - halfOpeningAngle;
          }
        } else if (r > 0 && std::abs(r) >= x) {
          if (atMinusZ) {
            displacement = -pt.halfZ() + delta;
            startPhi = 270. - halfOpeningAngle;
            h = pt.y1();
          } else {
            displacement = pt.halfZ() - delta;
            startPhi = 90. - halfOpeningAngle;
            h = pt.y2();
          }
        } else {
          throw cms::Exception("Check parameters of the PseudoTrap! name=" + name);
        }

        std::unique_ptr<TGeoShape> trap(
            new TGeoTrd2(name.c_str(), pt.x1() / cm, pt.x2() / cm, pt.y1() / cm, pt.y2() / cm, pt.halfZ() / cm));

        std::unique_ptr<TGeoShape> tubs(new TGeoTubeSeg(name.c_str(),
                                                        0.,
                                                        std::abs(r) / cm,  // radius cannot be negative!!!
                                                        h / cm,
                                                        startPhi,
                                                        startPhi + halfOpeningAngle * 2.));
        if (intersec) {
          TGeoSubtraction* sub = new TGeoSubtraction(
              trap.release(), tubs.release(), nullptr, createPlacement(s_rot, DDTranslation(0., 0., displacement)));
          rSolid = new TGeoCompositeShape(iName.c_str(), sub);
        } else {
          std::unique_ptr<TGeoShape> box(new TGeoBBox(1.1 * x / cm, 1.1 * h / cm, sqrt(r * r - x * x) / cm));

          TGeoSubtraction* sub = new TGeoSubtraction(
              tubs.release(), box.release(), nullptr, createPlacement(s_rot, DDTranslation(0., 0., 0.)));

          std::unique_ptr<TGeoShape> tubicCap(new TGeoCompositeShape(iName.c_str(), sub));

          TGeoUnion* boolS = new TGeoUnion(
              trap.release(), tubicCap.release(), nullptr, createPlacement(s_rot, DDTranslation(0., 0., displacement)));

          rSolid = new TGeoCompositeShape(iName.c_str(), boolS);
        }

        break;
      }
      case DDSolidShape::ddtorus: {
        DDTorus solid(iSolid);
        rSolid = new TGeoTorus(iName.c_str(),
                               solid.rTorus() / cm,
                               solid.rMin() / cm,
                               solid.rMax() / cm,
                               solid.startPhi() / deg,
                               solid.deltaPhi() / deg);
        break;
      }
      case DDSolidShape::ddsubtraction: {
        DDBooleanSolid boolSolid(iSolid);
        if (!boolSolid) {
          throw cms::Exception("GeomConvert") << "conversion to DDBooleanSolid failed";
        }

        std::string nameA = m_fullname ? boolSolid.solidA().name().fullname() : boolSolid.solidA().name().name();
        std::string nameB = m_fullname ? boolSolid.solidB().name().fullname() : boolSolid.solidB().name().name();
        std::unique_ptr<TGeoShape> left(createShape(nameA, boolSolid.solidA()));
        std::unique_ptr<TGeoShape> right(createShape(nameB, boolSolid.solidB()));
        if (nullptr != left.get() && nullptr != right.get()) {
          TGeoSubtraction* sub =
              new TGeoSubtraction(left.release(),
                                  right.release(),
                                  nullptr,
                                  createPlacement(boolSolid.rotation().matrix(), boolSolid.translation()));
          rSolid = new TGeoCompositeShape(iName.c_str(), sub);
        }
        break;
      }
      case DDSolidShape::ddtrunctubs: {
        DDTruncTubs tt(iSolid);
        if (!tt) {
          throw cms::Exception("GeomConvert") << "conversion to DDTruncTubs failed";
        }
        double rIn(tt.rIn());
        double rOut(tt.rOut());
        double zHalf(tt.zHalf());
        double startPhi(tt.startPhi());
        double deltaPhi(tt.deltaPhi());
        double cutAtStart(tt.cutAtStart());
        double cutAtDelta(tt.cutAtDelta());
        bool cutInside(bool(tt.cutInside()));
        std::string name = m_fullname ? tt.name().fullname() : tt.name().name();

        // check the parameters
        if (rIn <= 0 || rOut <= 0 || cutAtStart <= 0 || cutAtDelta <= 0) {
          std::string s = "TruncTubs " + name + ": 0 <= rIn,cutAtStart,rOut,cutAtDelta,rOut violated!";
          throw cms::Exception(s);
        }
        if (rIn >= rOut) {
          std::string s = "TruncTubs " + name + ": rIn<rOut violated!";
          throw cms::Exception(s);
        }
        if (startPhi != 0.) {
          std::string s = "TruncTubs " + name + ": startPhi != 0 not supported!";
          throw cms::Exception(s);
        }

        startPhi = 0.;
        double r(cutAtStart);
        double R(cutAtDelta);

        // Note: startPhi is always 0.0
        std::unique_ptr<TGeoShape> tubs(
            new TGeoTubeSeg(name.c_str(), rIn / cm, rOut / cm, zHalf / cm, startPhi, deltaPhi / deg));

        double boxX(rOut), boxY(rOut);  // exaggerate dimensions - does not matter, it's subtracted!

        // width of the box > width of the tubs
        double boxZ(1.1 * zHalf);

        // angle of the box w.r.t. tubs
        double cath = r - R * cos(deltaPhi);
        double hypo = sqrt(r * r + R * R - 2. * r * R * cos(deltaPhi));
        double cos_alpha = cath / hypo;
        double alpha = -acos(cos_alpha);

        // rotationmatrix of box w.r.t. tubs
        TGeoRotation rot;
        rot.RotateX(90);
        rot.RotateZ(alpha / deg);

        // center point of the box
        double xBox;
        if (!cutInside) {
          xBox = r + boxX / sin(fabs(alpha));
        } else {
          xBox = -(boxX / sin(fabs(alpha)) - r);
        }
        std::unique_ptr<TGeoShape> box(new TGeoBBox(name.c_str(), boxX / cm, boxZ / cm, boxY / cm));

        TGeoTranslation trans(xBox / cm, 0., 0.);

        TGeoSubtraction* sub =
            new TGeoSubtraction(tubs.release(), box.release(), nullptr, new TGeoCombiTrans(trans, rot));

        rSolid = new TGeoCompositeShape(iName.c_str(), sub);
        break;
      }
      case DDSolidShape::ddunion: {
        DDBooleanSolid boolSolid(iSolid);
        if (!boolSolid) {
          throw cms::Exception("GeomConvert") << "conversion to DDBooleanSolid failed";
        }

        std::string nameA = m_fullname ? boolSolid.solidA().name().fullname() : boolSolid.solidA().name().name();
        std::string nameB = m_fullname ? boolSolid.solidB().name().fullname() : boolSolid.solidB().name().name();
        std::unique_ptr<TGeoShape> left(createShape(nameA, boolSolid.solidA()));
        std::unique_ptr<TGeoShape> right(createShape(nameB, boolSolid.solidB()));
        //DEBUGGING
        //break;
        if (nullptr != left.get() && nullptr != right.get()) {
          TGeoUnion* boolS = new TGeoUnion(left.release(),
                                           right.release(),
                                           nullptr,
                                           createPlacement(boolSolid.rotation().matrix(), boolSolid.translation()));
          rSolid = new TGeoCompositeShape(iName.c_str(), boolS);
        }
        break;
      }
      case DDSolidShape::ddintersection: {
        DDBooleanSolid boolSolid(iSolid);
        if (!boolSolid) {
          throw cms::Exception("GeomConvert") << "conversion to DDBooleanSolid failed";
        }

        std::string nameA = m_fullname ? boolSolid.solidA().name().fullname() : boolSolid.solidA().name().name();
        std::string nameB = m_fullname ? boolSolid.solidB().name().fullname() : boolSolid.solidB().name().name();
        std::unique_ptr<TGeoShape> left(createShape(nameA, boolSolid.solidA()));
        std::unique_ptr<TGeoShape> right(createShape(nameB, boolSolid.solidB()));
        if (nullptr != left.get() && nullptr != right.get()) {
          TGeoIntersection* boolS =
              new TGeoIntersection(left.release(),
                                   right.release(),
                                   nullptr,
                                   createPlacement(boolSolid.rotation().matrix(), boolSolid.translation()));
          rSolid = new TGeoCompositeShape(iName.c_str(), boolS);
        }
        break;
      }
      case DDSolidShape::ddellipticaltube: {
        DDEllipticalTube eSolid(iSolid);
        if (!eSolid) {
          throw cms::Exception("GeomConvert") << "conversion to DDEllipticalTube failed";
        }
        rSolid = new TGeoEltu(iName.c_str(), params[0] / cm, params[1] / cm, params[2] / cm);
        break;
      }
      default:
        break;
    }
    nameToShape_[iName] = rSolid;
  }
  if (rSolid == nullptr) {
    std::cerr << "COULD NOT MAKE " << iName << " of a shape " << iSolid << std::endl;
  }

  LogDebug("TGeoMgrFromDdd::createShape") << "solid " << iName << " has been created.";

  return rSolid;
}

TGeoVolume* TGeoMgrFromDdd::createVolume(const std::string& iName, const DDSolid& iSolid, const DDMaterial& iMaterial) {
  TGeoVolume* v = nameToVolume_[iName];
  if (v == nullptr) {
    TGeoShape* solid =
        m_fullname ? createShape(iSolid.name().fullname(), iSolid) : createShape(iSolid.name().name(), iSolid);
    std::string mat_name = m_fullname ? iMaterial.name().fullname() : iMaterial.name().name();
    TGeoMedium* geo_med = nameToMedium_[mat_name];
    if (geo_med == nullptr) {
      TGeoMaterial* geo_mat = createMaterial(iMaterial);
      geo_med = new TGeoMedium(mat_name.c_str(), 0, geo_mat);
      nameToMedium_[mat_name] = geo_med;
    }
    if (solid) {
      v = new TGeoVolume(iName.c_str(), solid, geo_med);
    }
    nameToVolume_[iName] = v;
  }
  return v;
}

TGeoMaterial* TGeoMgrFromDdd::createMaterial(const DDMaterial& iMaterial) {
  std::string mat_name = m_fullname ? iMaterial.name().fullname() : iMaterial.name().name();
  TGeoMaterial* mat = nameToMaterial_[mat_name];

  if (mat == nullptr) {
    if (iMaterial.noOfConstituents() > 0) {
      TGeoMixture* mix = new TGeoMixture(mat_name.c_str(), iMaterial.noOfConstituents(), iMaterial.density() * cm3 / g);
      for (int i = 0; i < iMaterial.noOfConstituents(); ++i) {
        mix->AddElement(createMaterial(iMaterial.constituent(i).first), iMaterial.constituent(i).second);
      }
      mat = mix;
    } else {
      mat = new TGeoMaterial(mat_name.c_str(), iMaterial.a() * mole / g, iMaterial.z(), iMaterial.density() * cm3 / g);
    }
    nameToMaterial_[mat_name] = mat;
  }

  return mat;
}

//
// const member functions
//

//
// static member functions
//
DEFINE_FWK_EVENTSETUP_MODULE(TGeoMgrFromDdd);
