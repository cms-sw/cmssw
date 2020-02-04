/*
//\class GEMGeometryBuilder

 Description: GEM Geometry builder from DD and DD4HEP
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  27 Jan 2020 
*/
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include <DetectorDescription/DDCMS/interface/DDFilteredView.h>
#include <DetectorDescription/DDCMS/interface/DDCompactView.h>

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "Geometry/MuonNumbering/interface/DD4hep_GEMNumberingScheme.h"

#include "DataFormats/Math/interface/CMSUnits.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <algorithm>
#include <iostream>
#include <string>

using namespace cms_units::operators;

GEMGeometryBuilderFromDDD::GEMGeometryBuilderFromDDD() {}

GEMGeometryBuilderFromDDD::~GEMGeometryBuilderFromDDD() {}

// DDD
void GEMGeometryBuilderFromDDD::build(GEMGeometry& theGeometry,
                                      const DDCompactView* cview,
                                      const MuonDDDConstants& muonConstants) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapGEM";

  // Asking only for the MuonGEM's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fv(*cview, filter);

  LogDebug("GEMGeometryBuilderFromDDD") << "Building the geometry service";
  LogDebug("GEMGeometryBuilderFromDDD") << "About to run through the GEM structure\n"
                                        << " First logical part " << fv.logicalPart().name().name();

  bool doSuper = fv.firstChild();

  LogDebug("GEMGeometryBuilderFromDDD") << "doSuperChamber = " << doSuper;
  // loop over superchambers
  std::vector<GEMSuperChamber*> superChambers;

  while (doSuper) {
    // getting chamber id from eta partitions
    fv.firstChild();
    fv.firstChild();

    MuonDDDNumbering mdddnumCh(muonConstants);
    GEMNumberingScheme gemNumCh(muonConstants);
    int rawidCh = gemNumCh.baseNumberToUnitNumber(mdddnumCh.geoHistoryToBaseNumber(fv.geoHistory()));
    GEMDetId detIdCh = GEMDetId(rawidCh);

    // back to chambers

    fv.parent();
    fv.parent();

    // currently there is no superchamber in the geometry
    // only 2 chambers are present separated by a gap.
    // making superchamber out of the first chamber layer including the gap between chambers
    if (detIdCh.layer() == 1) {  // only make superChambers when doing layer 1

      GEMSuperChamber* gemSuperChamber = buildSuperChamber(fv, detIdCh);
      superChambers.push_back(gemSuperChamber);
    }

    GEMChamber* gemChamber = buildChamber(fv, detIdCh);

    // loop over chambers
    // only 1 chamber
    bool doChambers = fv.firstChild();
    bool loopExecuted = false;

    while (doChambers) {
      loopExecuted = true;

      // loop over GEMEtaPartitions
      bool doEtaPart = fv.firstChild();

      while (doEtaPart) {
        MuonDDDNumbering mdddnum(muonConstants);
        GEMNumberingScheme gemNum(muonConstants);
        int rawid = gemNum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
        GEMDetId detId = GEMDetId(rawid);
        GEMEtaPartition* etaPart = buildEtaPartition(fv, detId);
        gemChamber->add(etaPart);
        theGeometry.add(etaPart);
        doEtaPart = fv.nextSibling();
      }

      fv.parent();

      theGeometry.add(gemChamber);

      doChambers = fv.nextSibling();
    }
    fv.parent();

    doSuper = fv.nextSibling();

    if (!loopExecuted)
      delete gemChamber;
  }

  // construct the regions, stations and rings.
  for (int re = -1; re <= 1; re = re + 2) {
    GEMRegion* region = new GEMRegion(re);
    for (int st = 1; st <= GEMDetId::maxStationId; ++st) {
      GEMStation* station = new GEMStation(re, st);
      std::string sign(re == -1 ? "-" : "");
      std::string name("GE" + sign + std::to_string(st) + "/1");
      station->setName(name);
      for (int ri = 1; ri <= 1; ++ri) {
        GEMRing* ring = new GEMRing(re, st, ri);
        for (auto superChamber : superChambers) {
          const GEMDetId detId(superChamber->id());
          if (detId.region() != re || detId.station() != st || detId.ring() != ri)
            continue;

          superChamber->add(
              theGeometry.chamber(GEMDetId(detId.region(), detId.ring(), detId.station(), 1, detId.chamber(), 0)));
          superChamber->add(
              theGeometry.chamber(GEMDetId(detId.region(), detId.ring(), detId.station(), 2, detId.chamber(), 0)));
          ring->add(superChamber);
          theGeometry.add(superChamber);
          LogDebug("GEMGeometryBuilderFromDDD") << "Adding super chamber " << detId << " to ring: "
                                                << "re " << re << " st " << st << " ri " << ri << std::endl;
        }
        LogDebug("GEMGeometryBuilderFromDDD") << "Adding ring " << ri << " to station "
                                              << "re " << re << " st " << st << std::endl;
        station->add(ring);
        theGeometry.add(ring);
      }
      LogDebug("GEMGeometryBuilderFromDDD") << "Adding station " << st << " to region " << re << std::endl;
      region->add(station);
      theGeometry.add(station);
    }
    LogDebug("GEMGeometryBuilderFromDDD") << "Adding region " << re << " to the geometry " << std::endl;

    theGeometry.add(region);
  }
}

GEMSuperChamber* GEMGeometryBuilderFromDDD::buildSuperChamber(DDFilteredView& fv, GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromDDD") << "buildSuperChamber " << fv.logicalPart().name().name() << " " << detId
                                        << std::endl;

  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  std::vector<double> dpar = solid.solidA().parameters();

  double dy = geant_units::operators::convertMmToCm(dpar[0]);   //length is along local Y
  double dz = geant_units::operators::convertMmToCm(dpar[3]);   // thickness is long local Z
  double dx1 = geant_units::operators::convertMmToCm(dpar[4]);  // bottom width is along local X
  double dx2 = geant_units::operators::convertMmToCm(dpar[8]);  // top width is along local X

  const int nch = 2;
  const double chgap = 2.105;

  dpar = solid.solidB().parameters();

  dz += geant_units::operators::convertMmToCm(dpar[3]);  // chamber thickness
  dz *= nch;                                             // 2 chambers in superchamber
  dz += chgap;                                           // gap between chambers
  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

  LogDebug("GEMGeometryBuilderFromDDD") << "size " << dx1 << " " << dx2 << " " << dy << " " << dz << std::endl;

  GEMSuperChamber* superChamber = new GEMSuperChamber(detId.superChamberId(), surf);
  return superChamber;
}

GEMChamber* GEMGeometryBuilderFromDDD::buildChamber(DDFilteredView& fv, GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromDDD") << "buildChamber " << fv.logicalPart().name().name() << " " << detId
                                        << std::endl;

  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  std::vector<double> dpar = solid.solidA().parameters();

  double dy = geant_units::operators::convertMmToCm(dpar[0]);   //length is along local Y
  double dz = geant_units::operators::convertMmToCm(dpar[3]);   // thickness is long local Z
  double dx1 = geant_units::operators::convertMmToCm(dpar[4]);  // bottom width is along local X
  double dx2 = geant_units::operators::convertMmToCm(dpar[8]);  // top width is along local X
  dpar = solid.solidB().parameters();
  dz += geant_units::operators::convertMmToCm(dpar[3]);  // chamber thickness

  bool isOdd = detId.chamber() % 2;

  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

  LogDebug("GEMGeometryBuilderFromDDD") << "size " << dx1 << " " << dx2 << " " << dy << " " << dz << std::endl;

  GEMChamber* chamber = new GEMChamber(detId.chamberId(), surf);
  return chamber;
}

GEMEtaPartition* GEMGeometryBuilderFromDDD::buildEtaPartition(DDFilteredView& fv, GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromDDD") << "buildEtaPartition " << fv.logicalPart().name().name() << " " << detId
                                        << std::endl;

  // EtaPartition specific parameter (nstrips and npads)
  DDValue numbOfStrips("nStrips");
  DDValue numbOfPads("nPads");
  std::vector<const DDsvalues_type*> specs(fv.specifics());
  std::vector<const DDsvalues_type*>::iterator is = specs.begin();
  double nStrips = 0., nPads = 0.;
  for (; is != specs.end(); is++) {
    if (DDfetch(*is, numbOfStrips))
      nStrips = numbOfStrips.doubles()[0];
    if (DDfetch(*is, numbOfPads))
      nPads = numbOfPads.doubles()[0];
  }
  LogDebug("GEMGeometryBuilderFromDDD") << ((nStrips == 0.) ? ("No nStrips found!!")
                                                            : ("Number of strips: " + std::to_string(nStrips)));
  LogDebug("GEMGeometryBuilderFromDDD") << ((nPads == 0.) ? ("No nPads found!!")
                                                          : ("Number of pads: " + std::to_string(nPads)));

  // EtaPartition specific parameter (size)
  std::vector<double> dpar = fv.logicalPart().solid().parameters();

  double be = geant_units::operators::convertMmToCm(dpar[4]);  // half bottom edge
  double te = geant_units::operators::convertMmToCm(dpar[8]);  // half top edge
  double ap = geant_units::operators::convertMmToCm(dpar[0]);  // half apothem
  double ti = 0.4;                                             // half thickness

  std::vector<float> pars;
  pars.emplace_back(be);
  pars.emplace_back(te);
  pars.emplace_back(ap);
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(be, te, ap, ti), isOdd));
  std::string name = fv.logicalPart().name().name();
  GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, name, pars);

  LogDebug("GEMGeometryBuilderFromDDD") << "size " << be << " " << te << " " << ap << " " << ti << std::endl;

  GEMEtaPartition* etaPartition = new GEMEtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

GEMGeometryBuilderFromDDD::RCPBoundPlane GEMGeometryBuilderFromDDD::boundPlane(const DDFilteredView& fv,
                                                                               Bounds* bounds,
                                                                               bool isOddChamber) const {
  // extract the position
  const DDTranslation& trans(fv.translation());
  const Surface::PositionType posResult(float(trans.x() / cm), float(trans.y() / cm), float(trans.z() / cm));

  // now the rotation
  const DDRotationMatrix& rotation = fv.rotation();
  DD3Vector x, y, z;
  rotation.GetComponents(x, y, z);

  Surface::RotationType rotResult(float(x.X()),
                                  float(x.Y()),
                                  float(x.Z()),
                                  float(y.X()),
                                  float(y.Y()),
                                  float(y.Z()),
                                  float(z.X()),
                                  float(z.Y()),
                                  float(z.Z()));

  //Change of axes for the forward
  Basic3DVector<float> newX(1., 0., 0.);
  Basic3DVector<float> newY(0., 0., -1.);
  Basic3DVector<float> newZ(0., 1., 0.);

  rotResult.rotateAxes(newX, newY, newZ);

  return RCPBoundPlane(new BoundPlane(posResult, rotResult, bounds));
}

// DD4HEP

void GEMGeometryBuilderFromDDD::build(GEMGeometry& theGeometry,
                                      const cms::DDCompactView* cview,
                                      const cms::MuonNumbering& muonConstants) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapGEM";
  cms::DDFilteredView fv(cview->detector(), cview->detector()->worldVolume());
  cms::DDSpecParRefs refs;
  const cms::DDSpecParRegistry& mypar = cview->specpars();
  mypar.filter(refs, attribute, value);
  fv.mergedSpecifics(refs);

  bool doChambers = fv.firstChild();

  // loop over superchambers
  std::vector<GEMSuperChamber*> superChambers;

  while (doChambers) {
    MuonBaseNumber mbn = muonConstants.geoHistoryToBaseNumber(fv.history());
    cms::GEMNumberingScheme gemnum(muonConstants.values());
    gemnum.baseNumberToUnitNumber(mbn);
    GEMDetId detIdCh = GEMDetId(gemnum.getDetId());

    if (detIdCh.layer() == 1) {  // only make superChambers when doing layer 1
      GEMSuperChamber* gemSuperChamber = buildSuperChamber(fv, detIdCh);
      superChambers.push_back(gemSuperChamber);
    }

    GEMChamber* gemChamber = buildChamber(fv, detIdCh);

    fv.down();
    fv.down();
    bool doEtaPart = fv.nextSibling();
    while (doEtaPart) {
      MuonBaseNumber mbn = muonConstants.geoHistoryToBaseNumber(fv.history());
      cms::GEMNumberingScheme gemnum(muonConstants.values());
      gemnum.baseNumberToUnitNumber(mbn);
      GEMDetId detId = GEMDetId(gemnum.getDetId());
      GEMEtaPartition* etaPart = buildEtaPartition(fv, detId);

      gemChamber->add(etaPart);

      theGeometry.add(etaPart);
      doEtaPart = fv.sibling();
    }
    theGeometry.add(gemChamber);
    fv.up();
    fv.up();
    fv.up();

    doChambers = fv.firstChild();

  }  // close while Chambers

  // construct the regions, stations and rings.
  for (int re = -1; re <= 1; re = re + 2) {
    GEMRegion* region = new GEMRegion(re);
    for (int st = 1; st <= GEMDetId::maxStationId; ++st) {
      GEMStation* station = new GEMStation(re, st);
      std::string sign(re == -1 ? "-" : "");
      std::string name("GE" + sign + std::to_string(st) + "/1");
      station->setName(name);
      for (int ri = 1; ri <= 1; ++ri) {
        GEMRing* ring = new GEMRing(re, st, ri);
        for (auto superChamber : superChambers) {
          const GEMDetId detId(superChamber->id());
          if (detId.region() != re || detId.station() != st || detId.ring() != ri)
            continue;

          superChamber->add(
              theGeometry.chamber(GEMDetId(detId.region(), detId.ring(), detId.station(), 1, detId.chamber(), 0)));
          superChamber->add(
              theGeometry.chamber(GEMDetId(detId.region(), detId.ring(), detId.station(), 2, detId.chamber(), 0)));

          ring->add(superChamber);
          theGeometry.add(superChamber);

          LogDebug("GEMGeometryBuilderFromDDD") << "Adding super chamber " << detId << " to ring: "
                                                << "re " << re << " st " << st << " ri " << ri << std::endl;
        }
        LogDebug("GEMGeometryBuilderFromDDD") << "Adding ring " << ri << " to station "
                                              << "re " << re << " st " << st << std::endl;

        station->add(ring);
        theGeometry.add(ring);
      }
      LogDebug("GEMGeometryBuilderFromDDD") << "Adding station " << st << " to region " << re << std::endl;

      region->add(station);
      theGeometry.add(station);
    }
    LogDebug("GEMGeometryBuilderFromDDD") << "Adding region " << re << " to the geometry " << std::endl;

    theGeometry.add(region);
  }
}

GEMSuperChamber* GEMGeometryBuilderFromDDD::buildSuperChamber(cms::DDFilteredView& fv, GEMDetId detId) const {
  cms::DDSolid solid(fv.solid());
  auto solidA = solid.solidA();
  std::vector<double> dpar = solidA.dimensions();

  double dy = dpar[3];   //length is along local Y
  double dz = dpar[2];   // thickness is long local Z
  double dx1 = dpar[0];  // bottom width is along local X
  double dx2 = dpar[1];  // top width is along loc

  auto solidB = solid.solidB();
  dpar = solidB.dimensions();
  const int nch = 2;
  const double chgap = 2.105;

  dz += dpar[2];  // chamber thickness
  dz *= nch;      // 2 chambers in superchamber
  dz += chgap;    // gap between chambers

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

  GEMSuperChamber* superChamber = new GEMSuperChamber(detId.superChamberId(), surf);
  return superChamber;
}

GEMChamber* GEMGeometryBuilderFromDDD::buildChamber(cms::DDFilteredView& fv, GEMDetId detId) const {
  cms::DDSolid solid(fv.solid());
  auto solidA = solid.solidA();
  std::vector<double> dpar = solidA.dimensions();

  double dy = dpar[3];   //length is along local Y
  double dz = dpar[2];   // thickness is long local Z
  double dx1 = dpar[0];  // bottom width is along local X
  double dx2 = dpar[1];  // top width is along local X

  auto solidB = solid.solidB();
  dpar = solidB.dimensions();

  dz += dpar[2];  // chamber thickness

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

  GEMChamber* chamber = new GEMChamber(detId.chamberId(), surf);
  return chamber;
}

GEMEtaPartition* GEMGeometryBuilderFromDDD::buildEtaPartition(cms::DDFilteredView& fv, GEMDetId detId) const {
  // EtaPartition specific parameter (nstrips and npads)

  auto nStrips = fv.get<double>("nStrips");
  auto nPads = fv.get<double>("nPads");

  // EtaPartition specific parameter (size)

  std::vector<double> dpar = fv.parameters();

  double ti = 0.4;  // half thickness

  const std::vector<float> pars{float(dpar[0]), float(dpar[1]), float(dpar[3]), float(nStrips), float(nPads)};

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dpar[0], dpar[1], dpar[3], ti), isOdd));

  std::string_view name = fv.name();

  GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, std::string(name), pars);

  GEMEtaPartition* etaPartition = new GEMEtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

GEMGeometryBuilderFromDDD::RCPBoundPlane GEMGeometryBuilderFromDDD::boundPlane(const cms::DDFilteredView& fv,
                                                                               Bounds* bounds,
                                                                               bool isOddChamber) const {
  // extract the position
  const Double_t* tran = fv.trans();
  Surface::PositionType posResult(tran[0], tran[1], tran[2]);

  // now the rotation
  DDRotationMatrix rota;
  fv.rot(rota);
  DD3Vector x, y, z;
  rota.GetComponents(x, y, z);
  Surface::RotationType rotResult(float(x.X()),
                                  float(x.Y()),
                                  float(x.Z()),
                                  float(y.X()),
                                  float(y.Y()),
                                  float(y.Z()),
                                  float(z.X()),
                                  float(z.Y()),
                                  float(z.Z()));

  //Change of axes for the forward
  Basic3DVector<float> newX(1., 0., 0.);
  Basic3DVector<float> newY(0., 0., -1.);
  Basic3DVector<float> newZ(0., 1., 0.);

  rotResult.rotateAxes(newX, newY, newZ);

  return RCPBoundPlane(new BoundPlane(posResult, rotResult, bounds));
}
