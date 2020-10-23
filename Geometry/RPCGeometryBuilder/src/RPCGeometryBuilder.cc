/*
//\class RPCGeometryBuilder

 Description: RPC Geometry builder from DD & DD4hep
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
//          Modified: Fri, 29 May 2020, following what Sunanda Banerjee made in PR #29842 PR #29943 and Ianna Osborne in PR #29954    
*/
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilder.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/DDCMS/interface/DDFilteredView.h>
#include <DetectorDescription/DDCMS/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include <iostream>
#include <algorithm>
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DataFormats/Math/interface/GeantUnits.h"

using namespace cms_units::operators;

RPCGeometryBuilder::RPCGeometryBuilder() {}

// for DDD

std::unique_ptr<RPCGeometry> RPCGeometryBuilder::build(const DDCompactView* cview,
                                                       const MuonGeometryConstants& muonConstants) {
  const std::string attribute = "ReadOutName";
  const std::string value = "MuonRPCHits";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview, filter);
  return this->buildGeometry(fview, muonConstants);
}
// for DD4hep

std::unique_ptr<RPCGeometry> RPCGeometryBuilder::build(const cms::DDCompactView* cview,
                                                       const MuonGeometryConstants& muonConstants) {
  const std::string attribute = "ReadOutName";
  const std::string value = "MuonRPCHits";
  const cms::DDFilter filter(attribute, value);
  cms::DDFilteredView fview(*cview, filter);
  return this->buildGeometry(fview, muonConstants);
}
// for DDD

std::unique_ptr<RPCGeometry> RPCGeometryBuilder::buildGeometry(DDFilteredView& fview,
                                                               const MuonGeometryConstants& muonConstants) {
  LogDebug("RPCGeometryBuilder") << "Building the geometry service";
  std::unique_ptr<RPCGeometry> geometry = std::make_unique<RPCGeometry>();
  LogDebug("RPCGeometryBuilder") << "About to run through the RPC structure\n"
                                 << " First logical part " << fview.logicalPart().name().name();
  bool doSubDets = fview.firstChild();
  LogDebug("RPCGeometryBuilder") << "doSubDets = " << doSubDets;
  while (doSubDets) {
    LogDebug("RPCGeometryBuilder") << "start the loop";
    MuonGeometryNumbering mdddnum(muonConstants);
    LogDebug("RPCGeometryBuilder") << "Getting the Muon base Number";
    MuonBaseNumber mbn = mdddnum.geoHistoryToBaseNumber(fview.geoHistory());
    LogDebug("RPCGeometryBuilder") << "Start the Rpc Numbering Schema";
    RPCNumberingScheme rpcnum(muonConstants);
    LogDebug("RPCGeometryBuilder") << "Getting the Unit Number";
    const int detid = rpcnum.baseNumberToUnitNumber(mbn);
    LogDebug("RPCGeometryBuilder") << "Getting the RPC det Id " << detid;
    RPCDetId rpcid(detid);
    RPCDetId chid(rpcid.region(), rpcid.ring(), rpcid.station(), rpcid.sector(), rpcid.layer(), rpcid.subsector(), 0);
    LogDebug("RPCGeometryBuilder") << "The RPCDetid is " << rpcid;

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type*> specs(fview.specifics());
    int nStrips = 0;
    for (auto& spec : specs) {
      if (DDfetch(spec, numbOfStrips)) {
        nStrips = int(numbOfStrips.doubles()[0]);
      }
    }

    LogDebug("RPCGeometryBuilder") << ((nStrips == 0) ? ("No strip found!!") : (""));

    std::vector<double> dpar = fview.logicalPart().solid().parameters();
    std::string name = fview.logicalPart().name().name();
    DDTranslation tran = fview.translation();
    DDRotationMatrix rota = fview.rotation();
    Surface::PositionType pos(geant_units::operators::convertMmToCm(tran.x()),
                              geant_units::operators::convertMmToCm(tran.y()),
                              geant_units::operators::convertMmToCm(tran.z()));

    DD3Vector x, y, z;
    rota.GetComponents(x, y, z);
    Surface::RotationType rot(float(x.X()),
                              float(x.Y()),
                              float(x.Z()),
                              float(y.X()),
                              float(y.Y()),
                              float(y.Z()),
                              float(z.X()),
                              float(z.Y()),
                              float(z.Z()));

    RPCRollSpecs* rollspecs = nullptr;
    Bounds* bounds = nullptr;

    if (dpar.size() == 3) {
      const float width = geant_units::operators::convertMmToCm(dpar[0]);
      const float length = geant_units::operators::convertMmToCm(dpar[1]);
      const float thickness = geant_units::operators::convertMmToCm(dpar[2]);
      bounds = new RectangularPlaneBounds(width, length, thickness);
      const std::vector<float> pars = {width, length, float(numbOfStrips.doubles()[0])};

      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCBarrel, name, pars);
      LogDebug("RPCGeometryBuilder") << "Barrel " << name << " par " << width << " " << length << " " << thickness;

    } else {
      const float be = geant_units::operators::convertMmToCm(dpar[4]);
      const float te = geant_units::operators::convertMmToCm(dpar[8]);
      const float ap = geant_units::operators::convertMmToCm(dpar[0]);
      const float ti = 0.4;

      bounds = new TrapezoidalPlaneBounds(be, te, ap, ti);

      const std::vector<float> pars = {float(geant_units::operators::convertMmToCm(dpar[4])),
                                       float(geant_units::operators::convertMmToCm(dpar[8])),
                                       float(geant_units::operators::convertMmToCm(dpar[0])),
                                       float(numbOfStrips.doubles()[0])};
      LogDebug("RPCGeometryBuilder") << "Forward " << name << " par " << dpar[4] << " " << dpar[8] << " " << dpar[3]
                                     << " " << dpar[0];

      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCEndcap, name, pars);

      Basic3DVector<float> newX(1., 0., 0.);
      Basic3DVector<float> newY(0., 0., 1.);
      newY *= -1;
      Basic3DVector<float> newZ(0., 1., 0.);
      rot.rotateAxes(newX, newY, newZ);
    }
    LogDebug("RPCGeometryBuilder") << "   Number of strips " << nStrips;

    BoundPlane* bp = new BoundPlane(pos, rot, bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    RPCRoll* r = new RPCRoll(rpcid, surf, rollspecs);
    geometry->add(r);

    auto rls = chids.find(chid);
    if (rls == chids.end())
      rls = chids.insert(std::make_pair(chid, std::list<RPCRoll*>())).first;
    rls->second.emplace_back(r);

    doSubDets = fview.nextSibling();
  }
  for (auto& ich : chids) {
    const RPCDetId& chid = ich.first;
    const auto& rls = ich.second;

    BoundPlane* bp = nullptr;
    if (!rls.empty()) {
      const auto& refSurf = (*rls.begin())->surface();
      if (chid.region() == 0) {
        float corners[6] = {0, 0, 0, 0, 0, 0};
        for (auto rl : rls) {
          const double h2 = rl->surface().bounds().length() / 2;
          const double w2 = rl->surface().bounds().width() / 2;
          const auto x1y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-w2, -h2, 0)));
          const auto x2y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+w2, +h2, 0)));
          corners[0] = std::min(corners[0], x1y1AtRef.x());
          corners[1] = std::min(corners[1], x1y1AtRef.y());
          corners[2] = std::max(corners[2], x2y2AtRef.x());
          corners[3] = std::max(corners[3], x2y2AtRef.y());
          corners[4] = std::min(corners[4], x1y1AtRef.z());
          corners[5] = std::max(corners[5], x1y1AtRef.z());
        }
        const LocalPoint lpOfCentre((corners[0] + corners[2]) / 2, (corners[1] + corners[3]) / 2, 0);
        const auto gpOfCentre = refSurf.toGlobal(lpOfCentre);
        auto bounds = new RectangularPlaneBounds(
            (corners[2] - corners[0]) / 2, (corners[3] - corners[1]) / 2, (corners[5] - corners[4]) + 0.5);
        bp = new BoundPlane(gpOfCentre, refSurf.rotation(), bounds);
      } else {
        float cornersLo[3] = {0, 0, 0}, cornersHi[3] = {0, 0, 0};
        float cornersZ[2] = {0, 0};
        for (auto rl : rls) {
          const double h2 = rl->surface().bounds().length() / 2;
          const double w2 = rl->surface().bounds().width() / 2;
          const auto& topo = dynamic_cast<const TrapezoidalStripTopology&>(rl->specificTopology());
          const double r = topo.radius();
          const double wAtLo = w2 / r * (r - h2);
          const double wAtHi = w2 / r * (r + h2);

          const auto x1y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-wAtLo, -h2, 0)));
          const auto x2y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+wAtLo, -h2, 0)));
          const auto x1y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-wAtHi, +h2, 0)));
          const auto x2y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+wAtHi, +h2, 0)));

          cornersLo[0] = std::min(cornersLo[0], x1y1AtRef.x());
          cornersLo[1] = std::max(cornersLo[1], x2y1AtRef.x());
          cornersLo[2] = std::min(cornersLo[2], x1y1AtRef.y());

          cornersHi[0] = std::min(cornersHi[0], x1y2AtRef.x());
          cornersHi[1] = std::max(cornersHi[1], x2y2AtRef.x());
          cornersHi[2] = std::max(cornersHi[2], x1y2AtRef.y());

          cornersZ[0] = std::min(cornersZ[0], x1y1AtRef.z());
          cornersZ[1] = std::max(cornersZ[1], x1y1AtRef.z());
        }
        const LocalPoint lpOfCentre((cornersHi[0] + cornersHi[1]) / 2, (cornersLo[2] + cornersHi[2]) / 2, 0);
        const auto gpOfCentre = refSurf.toGlobal(lpOfCentre);
        auto bounds = new TrapezoidalPlaneBounds((cornersLo[1] - cornersLo[0]) / 2,
                                                 (cornersHi[1] - cornersHi[0]) / 2,
                                                 (cornersHi[2] - cornersLo[2]) / 2,
                                                 (cornersZ[1] - cornersZ[0]) + 0.5);
        bp = new BoundPlane(gpOfCentre, refSurf.rotation(), bounds);
      }
    }

    ReferenceCountingPointer<BoundPlane> surf(bp);
    RPCChamber* ch = new RPCChamber(chid, surf);
    for (auto rl : rls)
      ch->add(rl);
    geometry->add(ch);
  }

  return geometry;
}

// for DD4hep

std::unique_ptr<RPCGeometry> RPCGeometryBuilder::buildGeometry(cms::DDFilteredView& fview,
                                                               const MuonGeometryConstants& muonConstants) {
  std::unique_ptr<RPCGeometry> geometry = std::make_unique<RPCGeometry>();

  while (fview.firstChild()) {
    MuonGeometryNumbering mdddnum(muonConstants);
    RPCNumberingScheme rpcnum(muonConstants);
    int rawidCh = rpcnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fview.history()));
    RPCDetId rpcid = RPCDetId(rawidCh);

    RPCDetId chid(rpcid.region(), rpcid.ring(), rpcid.station(), rpcid.sector(), rpcid.layer(), rpcid.subsector(), 0);

    auto nStrips = fview.get<double>("nStrips");

    std::vector<double> dpar = fview.parameters();

    std::string_view name = fview.name();

    const Double_t* tran = fview.trans();

    DDRotationMatrix rota;
    fview.rot(rota);

    Surface::PositionType pos(tran[0], tran[1], tran[2]);

    DD3Vector x, y, z;
    rota.GetComponents(x, y, z);
    Surface::RotationType rot(float(x.X()),
                              float(x.Y()),
                              float(x.Z()),
                              float(y.X()),
                              float(y.Y()),
                              float(y.Z()),
                              float(z.X()),
                              float(z.Y()),
                              float(z.Z()));

    RPCRollSpecs* rollspecs = nullptr;
    Bounds* bounds = nullptr;

    if (dd4hep::isA<dd4hep::Box>(fview.solid())) {
      const float width = dpar[0];
      const float length = dpar[1];
      const float thickness = dpar[2];

      bounds = new RectangularPlaneBounds(width, length, thickness);

      const std::vector<float> pars = {width, length, float(nStrips)};

      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCBarrel, std::string(name), pars);

    } else {
      const float be = dpar[0];
      const float te = dpar[1];
      const float ap = dpar[3];
      const float ti = 0.4;

      bounds = new TrapezoidalPlaneBounds(be, te, ap, ti);
      const std::vector<float> pars = {float(dpar[0]), float(dpar[1]), float(dpar[3]), float(nStrips)};

      rollspecs = new RPCRollSpecs(GeomDetEnumerators::RPCEndcap, std::string(name), pars);

      Basic3DVector<float> newX(1., 0., 0.);
      Basic3DVector<float> newY(0., 0., 1.);
      newY *= -1;
      Basic3DVector<float> newZ(0., 1., 0.);
      rot.rotateAxes(newX, newY, newZ);
    }

    BoundPlane* bp = new BoundPlane(pos, rot, bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    RPCRoll* r = new RPCRoll(rpcid, surf, rollspecs);
    geometry->add(r);

    auto rls = chids.find(chid);
    if (rls == chids.end())
      rls = chids.insert(std::make_pair(chid, std::list<RPCRoll*>())).first;
    rls->second.emplace_back(r);
  }

  for (auto& ich : chids) {
    const RPCDetId& chid = ich.first;
    const auto& rls = ich.second;

    BoundPlane* bp = nullptr;
    if (!rls.empty()) {
      const auto& refSurf = (*rls.begin())->surface();
      if (chid.region() == 0) {
        float corners[6] = {0, 0, 0, 0, 0, 0};
        for (auto rl : rls) {
          const double h2 = rl->surface().bounds().length() / 2;
          const double w2 = rl->surface().bounds().width() / 2;
          const auto x1y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-w2, -h2, 0)));
          const auto x2y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+w2, +h2, 0)));
          corners[0] = std::min(corners[0], x1y1AtRef.x());
          corners[1] = std::min(corners[1], x1y1AtRef.y());
          corners[2] = std::max(corners[2], x2y2AtRef.x());
          corners[3] = std::max(corners[3], x2y2AtRef.y());

          corners[4] = std::min(corners[4], x1y1AtRef.z());
          corners[5] = std::max(corners[5], x1y1AtRef.z());
        }
        const LocalPoint lpOfCentre((corners[0] + corners[2]) / 2, (corners[1] + corners[3]) / 2, 0);
        const auto gpOfCentre = refSurf.toGlobal(lpOfCentre);
        auto bounds = new RectangularPlaneBounds(
            (corners[2] - corners[0]) / 2, (corners[3] - corners[1]) / 2, (corners[5] - corners[4]) + 0.5);
        bp = new BoundPlane(gpOfCentre, refSurf.rotation(), bounds);

      } else {
        float cornersLo[3] = {0, 0, 0}, cornersHi[3] = {0, 0, 0};
        float cornersZ[2] = {0, 0};
        for (auto rl : rls) {
          const double h2 = rl->surface().bounds().length() / 2;
          const double w2 = rl->surface().bounds().width() / 2;
          const auto& topo = dynamic_cast<const TrapezoidalStripTopology&>(rl->specificTopology());
          const double r = topo.radius();
          const double wAtLo = w2 / r * (r - h2);
          const double wAtHi = w2 / r * (r + h2);
          const auto x1y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-wAtLo, -h2, 0)));
          const auto x2y1AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+wAtLo, -h2, 0)));
          const auto x1y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(-wAtHi, +h2, 0)));
          const auto x2y2AtRef = refSurf.toLocal(rl->toGlobal(LocalPoint(+wAtHi, +h2, 0)));

          cornersLo[0] = std::min(cornersLo[0], x1y1AtRef.x());
          cornersLo[1] = std::max(cornersLo[1], x2y1AtRef.x());
          cornersLo[2] = std::min(cornersLo[2], x1y1AtRef.y());

          cornersHi[0] = std::min(cornersHi[0], x1y2AtRef.x());
          cornersHi[1] = std::max(cornersHi[1], x2y2AtRef.x());
          cornersHi[2] = std::max(cornersHi[2], x1y2AtRef.y());

          cornersZ[0] = std::min(cornersZ[0], x1y1AtRef.z());
          cornersZ[1] = std::max(cornersZ[1], x1y1AtRef.z());
        }
        const LocalPoint lpOfCentre((cornersHi[0] + cornersHi[1]) / 2, (cornersLo[2] + cornersHi[2]) / 2, 0);
        const auto gpOfCentre = refSurf.toGlobal(lpOfCentre);
        auto bounds = new TrapezoidalPlaneBounds((cornersLo[1] - cornersLo[0]) / 2,
                                                 (cornersHi[1] - cornersHi[0]) / 2,
                                                 (cornersHi[2] - cornersLo[2]) / 2,
                                                 (cornersZ[1] - cornersZ[0]) + 0.5);
        bp = new BoundPlane(gpOfCentre, refSurf.rotation(), bounds);
      }
    }

    ReferenceCountingPointer<BoundPlane> surf(bp);

    RPCChamber* ch = new RPCChamber(chid, surf);

    for (auto rl : rls)
      ch->add(rl);

    geometry->add(ch);
  }
  return geometry;
}
