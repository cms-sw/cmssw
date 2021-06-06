/*
//\class GEMGeometryBuilder

 Description: GEM Geometry builder from DD and DD4HEP
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
              Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
              Updated by Sunanda Banerjee (Fermilab) to make it working for dd4hep
              Updated:  7 August 2020 
*/
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilder.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include "DataFormats/Math/interface/GeantUnits.h"

#include <algorithm>
#include <iostream>
#include <string>

using namespace geant_units::operators;

//#define EDM_ML_DEBUG

GEMGeometryBuilder::GEMGeometryBuilder() {}

GEMGeometryBuilder::~GEMGeometryBuilder() {}

// DDD
void GEMGeometryBuilder::build(GEMGeometry& theGeometry,
                               const DDCompactView* cview,
                               const MuonGeometryConstants& muonConstants) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapGEM";

  // Asking only for the MuonGEM's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fv(*cview, filter);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "Building the geometry service";
  edm::LogVerbatim("Geometry") << "About to run through the GEM structure\n"
                               << " First logical part " << fv.logicalPart().name().name();
#endif
  bool doSuper = fv.firstChild();

  MuonGeometryNumbering mdddnum(muonConstants);
  GEMNumberingScheme gemNum(muonConstants);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "doSuperChamber = " << doSuper << " with " << fv.geoHistory() << " Levels "
                               << mdddnum.geoHistoryToBaseNumber(fv.geoHistory()).getLevels();
  ;
#endif
  // loop over superchambers
  std::vector<GEMSuperChamber*> superChambers;
  while (doSuper) {
    // getting chamber id from eta partitions
    fv.firstChild();
    fv.firstChild();

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Geometry") << "MuonGeometry 1 " << fv.geoHistory() << " Levels "
                                 << mdddnum.geoHistoryToBaseNumber(fv.geoHistory()).getLevels();
#endif
    int rawidCh = gemNum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
    GEMDetId detIdCh = GEMDetId(rawidCh);

    // back to chambers

    fv.parent();
    fv.parent();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Geometry") << "MuonGeometry 2 " << fv.geoHistory() << " Levels "
                                 << mdddnum.geoHistoryToBaseNumber(fv.geoHistory()).getLevels();
    gemNum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
#endif
    // currently there is no superchamber in the geometry
    // only 2 chambers are present separated by a gap.
    // making superchamber out of the first chamber layer including the gap between chambers
    if (detIdCh.layer() == 1) {  // only make superChambers when doing layer 1
      GEMSuperChamber* gemSuperChamber = buildSuperChamber(fv, detIdCh);
      superChambers.push_back(gemSuperChamber);
    }
    GEMChamber* gemChamber = ((detIdCh.station() == GEMDetId::minStationId0) ? nullptr : buildChamber(fv, detIdCh));

    // loop over chambers
    // only 1 chamber
    bool doChambers = fv.firstChild();
    bool loopExecuted = false;

    while (doChambers) {
      loopExecuted = true;

      if (detIdCh.station() == GEMDetId::minStationId0) {
        fv.firstChild();
        int rawId = gemNum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
        GEMDetId detId = GEMDetId(rawId);
        fv.parent();
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("Geometry") << "MuonGeometry 3 " << fv.geoHistory() << " Levels "
                                     << mdddnum.geoHistoryToBaseNumber(fv.geoHistory()).getLevels();
        gemNum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
#endif
        gemChamber = buildChamber(fv, detId);
      }

      // loop over GEMEtaPartitions
      bool doEtaPart = fv.firstChild();

      while (doEtaPart) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("Geometry") << "MuonGeometry 4 " << fv.geoHistory() << " Levels "
                                     << mdddnum.geoHistoryToBaseNumber(fv.geoHistory()).getLevels();
#endif
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

    if (!loopExecuted) {
      delete gemChamber;
    }
  }

  buildRegions(theGeometry, superChambers);
}

GEMSuperChamber* GEMGeometryBuilder::buildSuperChamber(DDFilteredView& fv, GEMDetId detId) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "buildSuperChamber " << fv.logicalPart().name().name() << " " << detId;
#endif
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  bool ge0Station = detId.station() == GEMDetId::minStationId0;
  std::vector<double> dpar = ge0Station ? solid.parameters() : solid.solidA().parameters();

  double dy = convertMmToCm(dpar[0]);   //length is along local Y
  double dz = convertMmToCm(dpar[3]);   // thickness is long local Z
  double dx1 = convertMmToCm(dpar[4]);  // bottom width is along local X
  double dx2 = convertMmToCm(dpar[8]);  // top width is along local X

  if (!ge0Station) {
    const int nch = 2;
    const double chgap = 2.105;

    dpar = solid.solidB().parameters();

    dz += convertMmToCm(dpar[3]);  // chamber thickness
    dz *= nch;                     // 2 chambers in superchamber
    dz += chgap;                   // gap between chambers
  }

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "size " << dx1 << " " << dx2 << " " << dy << " " << dz;
#endif
  GEMSuperChamber* superChamber = new GEMSuperChamber(detId.superChamberId(), surf);
  return superChamber;
}

GEMChamber* GEMGeometryBuilder::buildChamber(DDFilteredView& fv, GEMDetId detId) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "buildChamber " << fv.logicalPart().name().name() << " " << detId;
#endif
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  bool ge0Station = detId.station() == GEMDetId::minStationId0;
  std::vector<double> dpar = ge0Station ? solid.parameters() : solid.solidA().parameters();

  double dy = convertMmToCm(dpar[0]);   //length is along local Y
  double dz = convertMmToCm(dpar[3]);   // thickness is long local Z
  double dx1 = convertMmToCm(dpar[4]);  // bottom width is along local X
  double dx2 = convertMmToCm(dpar[8]);  // top width is along local X

  if (!ge0Station) {
    dpar = solid.solidB().parameters();
    dz += convertMmToCm(dpar[3]);  // chamber thickness
  }

  bool isOdd = ge0Station ? false : detId.chamber() % 2;

  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "size " << dx1 << " " << dx2 << " " << dy << " " << dz;
#endif
  GEMChamber* chamber = new GEMChamber(detId.chamberId(), surf);
  return chamber;
}

GEMEtaPartition* GEMGeometryBuilder::buildEtaPartition(DDFilteredView& fv, GEMDetId detId) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "buildEtaPartition " << fv.logicalPart().name().name() << " " << detId;
#endif
  // EtaPartition specific parameter (nstrips and npads)
  DDValue numbOfStrips("nStrips");
  DDValue numbOfPads("nPads");
  DDValue delPhi("dPhi");
  std::vector<const DDsvalues_type*> specs(fv.specifics());
  std::vector<const DDsvalues_type*>::iterator is = specs.begin();
  double nStrips = 0., nPads = 0., dPhi = 0.;
  for (; is != specs.end(); is++) {
    if (DDfetch(*is, numbOfStrips))
      nStrips = numbOfStrips.doubles()[0];
    if (DDfetch(*is, numbOfPads))
      nPads = numbOfPads.doubles()[0];
    if (DDfetch(*is, delPhi))
      dPhi = delPhi.doubles()[0];
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << ((nStrips == 0.) ? ("No nStrips found!!")
                                                   : ("Number of strips: " + std::to_string(nStrips)));
  edm::LogVerbatim("Geometry") << ((nPads == 0.) ? ("No nPads found!!") : ("Number of pads: " + std::to_string(nPads)));
#endif
  // EtaPartition specific parameter (size)
  std::vector<double> dpar = fv.logicalPart().solid().parameters();

  double be = convertMmToCm(dpar[4]);  // half bottom edge
  double te = convertMmToCm(dpar[8]);  // half top edge
  double ap = convertMmToCm(dpar[0]);  // half apothem
  double ti = 0.4;                     // half thickness

  std::vector<float> pars;
  pars.emplace_back(be);
  pars.emplace_back(te);
  pars.emplace_back(ap);
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);
  pars.emplace_back(dPhi);

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(be, te, ap, ti), isOdd));
  std::string name = fv.logicalPart().name().name();
  GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, name, pars);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "size " << be << " " << te << " " << ap << " " << ti;
#endif
  GEMEtaPartition* etaPartition = new GEMEtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

GEMGeometryBuilder::RCPBoundPlane GEMGeometryBuilder::boundPlane(const DDFilteredView& fv,
                                                                 Bounds* bounds,
                                                                 bool isOddChamber) const {
  // extract the position
  const DDTranslation& trans(fv.translation());
  const Surface::PositionType posResult(convertMmToCm(trans.x()), convertMmToCm(trans.y()), convertMmToCm(trans.z()));

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

void GEMGeometryBuilder::build(GEMGeometry& theGeometry,
                               const cms::DDCompactView* cview,
                               const MuonGeometryConstants& muonConstants) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapGEM";
  const cms::DDFilter filter(attribute, value);
  cms::DDFilteredView fv(*cview, filter);

  MuonGeometryNumbering mdddnum(muonConstants);
  GEMNumberingScheme gemNum(muonConstants);
  static constexpr uint32_t levelChamb = 7;
  int chamb(0), region(0);
  int theLevelPart = muonConstants.getValue("level");
  int theRingLevel = muonConstants.getValue("mg_ring") / theLevelPart;
  int theSectorLevel = muonConstants.getValue("mg_sector") / theLevelPart;
  std::vector<GEMSuperChamber*> superChambers;
  std::vector<GEMChamber*> chambers;

  while (fv.firstChild()) {
    const auto& history = fv.history();
    MuonBaseNumber num(mdddnum.geoHistoryToBaseNumber(history));
    GEMDetId detId(gemNum.baseNumberToUnitNumber(num));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Geometry") << fv.name() << " with " << history.tags.size() << " Levels and ID " << detId
                                 << " Mask " << std::hex << GEMDetId::chamberIdMask << std::dec << " and "
                                 << GEMDetId(((detId.rawId()) & GEMDetId::chamberIdMask)) << " Levels " << theRingLevel
                                 << ":" << theSectorLevel << ":" << history.tags.size() << ":" << fv.level();
    for (unsigned int k = 0; k < history.tags.size(); ++k)
      edm::LogVerbatim("Geometry") << "[" << k << "] Tag " << history.tags[k] << " Offset " << history.offsets[k]
                                   << " copy " << history.copyNos[k];
#endif

    if (detId.station() == GEMDetId::minStationId0) {
      if (num.getLevels() == theRingLevel) {
        if (detId.region() != region) {
          region = detId.region();
          chamb = 0;
        }
        ++chamb;
        detId = GEMDetId(detId.region(), detId.ring(), detId.station(), detId.layer(), chamb, 0);
        GEMSuperChamber* gemSuperChamber = buildSuperChamber(fv, detId);
        superChambers.emplace_back(gemSuperChamber);
      } else if (num.getLevels() == theSectorLevel) {
        GEMChamber* gemChamber = buildChamber(fv, detId);
        chambers.emplace_back(gemChamber);
      } else {
        GEMEtaPartition* etaPart = buildEtaPartition(fv, detId);
        theGeometry.add(etaPart);
      }
    } else {
      if (fv.level() == levelChamb) {
        if (detId.layer() == 1) {
          GEMSuperChamber* gemSuperChamber = buildSuperChamber(fv, detId);
          superChambers.emplace_back(gemSuperChamber);
        }
        GEMChamber* gemChamber = buildChamber(fv, detId);
        chambers.emplace_back(gemChamber);
      } else if (num.getLevels() > theSectorLevel) {
        GEMEtaPartition* etaPart = buildEtaPartition(fv, detId);
        theGeometry.add(etaPart);
      }
    }
  }

  auto& partitions = theGeometry.etaPartitions();
  for (auto& gemChamber : chambers) {
    uint32_t id0 = ((gemChamber->id().rawId()) & GEMDetId::chamberIdMask);
    for (auto& etaPart : partitions) {
      if (((etaPart->id().rawId()) & GEMDetId::chamberIdMask) == id0) {
        gemChamber->add(etaPart);
      }
    }
    theGeometry.add(gemChamber);
  }

  buildRegions(theGeometry, superChambers);
}

GEMSuperChamber* GEMGeometryBuilder::buildSuperChamber(cms::DDFilteredView& fv, GEMDetId detId) const {
  cms::DDSolid solid(fv.solid());
  auto solidA = solid.solidA();
  std::vector<double> dpar = solidA.dimensions();

  double dy = k_ScaleFromDD4Hep * dpar[3];   //length is along local Y
  double dz = k_ScaleFromDD4Hep * dpar[2];   // thickness is long local Z
  double dx1 = k_ScaleFromDD4Hep * dpar[0];  // bottom width is along local X
  double dx2 = k_ScaleFromDD4Hep * dpar[1];  // top width is along loc

  auto solidB = solid.solidB();
  dpar = solidB.dimensions();
  const int nch = 2;
  const double chgap = 2.105;

  dz += (k_ScaleFromDD4Hep * dpar[2]);  // chamber thickness
  dz *= nch;                            // 2 chambers in superchamber
  dz += chgap;                          // gap between chambers

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

  GEMSuperChamber* superChamber = new GEMSuperChamber(detId.superChamberId(), surf);
  return superChamber;
}

GEMChamber* GEMGeometryBuilder::buildChamber(cms::DDFilteredView& fv, GEMDetId detId) const {
  cms::DDSolid solid(fv.solid());
  auto solidA = solid.solidA();
  std::vector<double> dpar = solidA.dimensions();

  double dy = k_ScaleFromDD4Hep * dpar[3];   //length is along local Y
  double dz = k_ScaleFromDD4Hep * dpar[2];   // thickness is long local Z
  double dx1 = k_ScaleFromDD4Hep * dpar[0];  // bottom width is along local X
  double dx2 = k_ScaleFromDD4Hep * dpar[1];  // top width is along local X

  auto solidB = solid.solidB();
  dpar = solidB.dimensions();

  dz += (k_ScaleFromDD4Hep * dpar[2]);  // chamber thickness

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1, dx2, dy, dz), isOdd));

  GEMChamber* chamber = new GEMChamber(detId.chamberId(), surf);
  return chamber;
}

GEMEtaPartition* GEMGeometryBuilder::buildEtaPartition(cms::DDFilteredView& fv, GEMDetId detId) const {
  // EtaPartition specific parameter (nstrips and npads)

  auto nStrips = fv.get<double>("nStrips");
  auto nPads = fv.get<double>("nPads");
  auto dPhi = fv.get<double>("dPhi");
  // EtaPartition specific parameter (size)

  std::vector<double> dpar = fv.parameters();

  double ti = 0.4;  // half thickness

  const std::vector<float> pars{float(k_ScaleFromDD4Hep * dpar[0]),
                                float(k_ScaleFromDD4Hep * dpar[1]),
                                float(k_ScaleFromDD4Hep * dpar[3]),
                                float(nStrips),
                                float(nPads),
                                float(dPhi)};

  bool isOdd = detId.chamber() % 2;
  RCPBoundPlane surf(
      boundPlane(fv,
                 new TrapezoidalPlaneBounds(
                     k_ScaleFromDD4Hep * dpar[0], k_ScaleFromDD4Hep * dpar[1], k_ScaleFromDD4Hep * dpar[3], ti),
                 isOdd));

  std::string_view name = fv.name();

  GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, std::string(name), pars);

  GEMEtaPartition* etaPartition = new GEMEtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

GEMGeometryBuilder::RCPBoundPlane GEMGeometryBuilder::boundPlane(const cms::DDFilteredView& fv,
                                                                 Bounds* bounds,
                                                                 bool isOddChamber) const {
  // extract the position
  const Double_t* tran = fv.trans();
  Surface::PositionType posResult(
      k_ScaleFromDD4Hep * tran[0], k_ScaleFromDD4Hep * tran[1], k_ScaleFromDD4Hep * tran[2]);

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

void GEMGeometryBuilder::buildRegions(GEMGeometry& theGeometry, const std::vector<GEMSuperChamber*>& superChambers) {
  // construct the regions, stations and rings.
  for (int re = -1; re <= 1; re = re + 2) {
    GEMRegion* region = new GEMRegion(re);
    for (int st = GEMDetId::minStationId0; st <= GEMDetId::maxStationId; ++st) {
      bool ge0Station = st == GEMDetId::minStationId0;
      GEMStation* station = new GEMStation(re, st);
      std::string sign(re == -1 ? "-" : "");
      std::string suffix = ge0Station ? "" : "/1";
      std::string name = "GE" + sign + std::to_string(st) + suffix;
      station->setName(name);
      bool foundSuperChamber = false;
      for (int ri = 1; ri <= 1; ++ri) {
        GEMRing* ring = new GEMRing(re, st, ri);
        for (auto superChamber : superChambers) {
          const GEMDetId detId(superChamber->id());
          if (detId.region() != re || detId.station() != st || detId.ring() != ri)
            continue;

          foundSuperChamber = true;
          int nlayers = ge0Station ? GEMDetId::maxLayerId0 : GEMDetId::maxLayerId;

          // GEMDetId::minLayerId is to id the superchamber, so minLayerId+1 is the first layer
          for (int la = GEMDetId::minLayerId + 1; la <= nlayers; ++la) {
            GEMDetId chId(detId.region(), detId.ring(), detId.station(), la, detId.chamber(), 0);
            auto chamber = theGeometry.chamber(chId);
            if (!chamber) {
              edm::LogWarning("GEMGeometryBuilder") << "Missing chamber " << chId;
            }
            superChamber->add(chamber);
          }
          ring->add(superChamber);
          theGeometry.add(superChamber);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("Geometry") << "Adding super chamber " << detId << " to ring: "
                                       << "re " << re << " st " << st << " ri " << ri;
#endif
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("Geometry") << "Adding ring " << ri << " to station "
                                     << "re " << re << " st " << st;
#endif
        if (foundSuperChamber) {
          station->add(ring);
          theGeometry.add(ring);
        }
      }
      if (!foundSuperChamber) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("Geometry") << "No superchamber found: re:" << re << " st:" << st;
#endif
        delete station;
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("Geometry") << "Adding station " << st << " to region " << re;
#endif
        region->add(station);
        theGeometry.add(station);
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Geometry") << "Adding region " << re << " to the geometry ";
#endif
    theGeometry.add(region);
  }
}
