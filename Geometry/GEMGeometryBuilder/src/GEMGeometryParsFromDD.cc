/* Implementation of the  GEMGeometryParsFromDD Class
 *  Build the GEMGeometry from the DDD and DD4Hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
 *  Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Mon, 15 Feb 2021 
 *
 */
#include "Geometry/GEMGeometryBuilder/interface/GEMGeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include <iostream>
#include <algorithm>

GEMGeometryParsFromDD::GEMGeometryParsFromDD() {}

GEMGeometryParsFromDD::~GEMGeometryParsFromDD() {}

// DDD

void GEMGeometryParsFromDD::build(const DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rgeo) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapGEM";

  // Asking only for the MuonGEM's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fv(*cview, filter);

  this->buildGeometry(fv, muonConstants, rgeo);
}

void GEMGeometryParsFromDD::buildGeometry(DDFilteredView& fv,
                                          const MuonGeometryConstants& muonConstants,
                                          RecoIdealGeometry& rgeo) {
  LogDebug("GEMGeometryParsFromDD") << "Building the geometry service";
  LogDebug("GEMGeometryParsFromDD") << "About to run through the GEM structure\n"
                                    << " First logical part " << fv.logicalPart().name().name();

  edm::LogVerbatim("GEMGeometryParsFromDD") << "(0) GEMGeometryParsFromDD - DDD ";
  MuonGeometryNumbering muonDDDNumbering(muonConstants);
  GEMNumberingScheme gemNumbering(muonConstants);

  bool doSuper = fv.firstChild();

  LogDebug("GEMGeometryParsFromDD") << "doSuperChamber = " << doSuper;
  // loop over superchambers
  while (doSuper) {
    // getting chamber id from eta partitions
    fv.firstChild();
    fv.firstChild();
    GEMDetId detIdCh =
        GEMDetId(gemNumbering.baseNumberToUnitNumber(muonDDDNumbering.geoHistoryToBaseNumber(fv.geoHistory())));
    // back to chambers
    fv.parent();
    fv.parent();

    // currently there is no superchamber in the geometry
    // only 2 chambers are present separated by a gap.
    // making superchamber out of the first chamber layer including the gap between chambers
    if (detIdCh.layer() == 1) {  // only make superChambers when doing layer 1
      buildSuperChamber(fv, detIdCh, rgeo);
    }
    buildChamber(fv, detIdCh, rgeo);

    // loop over chambers
    // only 1 chamber
    bool doChambers = fv.firstChild();
    while (doChambers) {
      // loop over GEMEtaPartitions
      bool doEtaPart = fv.firstChild();
      while (doEtaPart) {
        GEMDetId detId =
            GEMDetId(gemNumbering.baseNumberToUnitNumber(muonDDDNumbering.geoHistoryToBaseNumber(fv.geoHistory())));
        buildEtaPartition(fv, detId, rgeo);

        doEtaPart = fv.nextSibling();
      }
      fv.parent();
      doChambers = fv.nextSibling();
    }
    fv.parent();
    doSuper = fv.nextSibling();
  }
}

void GEMGeometryParsFromDD::buildSuperChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo) {
  LogDebug("GEMGeometryParsFromDD") << "buildSuperChamber " << fv.logicalPart().name().name() << " " << detId
                                    << std::endl;

  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  std::vector<double> dpar = solid.solidA().parameters();

  GEMDetId gemid = detId.superChamberId();

  double dy = dpar[0];   //length is along local Y
  double dz = dpar[3];   // thickness is long local Z
  double dx1 = dpar[4];  // bottom width is along local X
  double dx2 = dpar[8];  // top width is along local X
  dpar = solid.solidB().parameters();

  dz += dpar[3];  // chamber thickness
  dz *= 2;        // 2 chambers in superchamber
  dz += 2.105;    // gap between chambers

  std::vector<double> pars{dx1, dx2, dy, dz};
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  LogDebug("GEMGeometryParsFromDD") << "dimension dx1 " << dx1 << ", dx2 " << dx2 << ", dy " << dy << ", dz " << dz;
  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(3) DDD, SuperChamber DetID " << gemid.rawId() << " Name " << fv.logicalPart().name().name() << " dx1 " << dx1
      << " dx2 " << dx2 << " dy " << dy << " dz " << dz;
  rgeo.insert(gemid.rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

void GEMGeometryParsFromDD::buildChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo) {
  LogDebug("GEMGeometryParsFromDD") << "buildChamber " << fv.logicalPart().name().name() << " " << detId << std::endl;

  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  std::vector<double> dpar = solid.solidA().parameters();

  double dy = dpar[0];   //length is along local Y
  double dz = dpar[3];   // thickness is long local Z
  double dx1 = dpar[4];  // bottom width is along local X
  double dx2 = dpar[8];  // top width is along local X
  dpar = solid.solidB().parameters();
  dz += dpar[3];  // chamber thickness

  GEMDetId gemid = detId.chamberId();

  std::vector<double> pars{dx1, dx2, dy, dz};
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  LogDebug("GEMGeometryParsFromDD") << "dimension dx1 " << dx1 << ", dx2 " << dx2 << ", dy " << dy << ", dz " << dz;
  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(4) DDD, Chamber DetID " << gemid.rawId() << " Name " << fv.logicalPart().name().name() << " dx1 " << dx1
      << " dx2 " << dx2 << " dy " << dy << " dz " << dz;
  rgeo.insert(gemid.rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

void GEMGeometryParsFromDD::buildEtaPartition(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo) {
  LogDebug("GEMGeometryParsFromDD") << "buildEtaPartition " << fv.logicalPart().name().name() << " " << detId
                                    << std::endl;

  // EtaPartition specific parameter (nstrips and npads)
  DDValue numbOfStrips("nStrips");
  DDValue numbOfPads("nPads");
  DDValue delPhi("dPhi");
  const std::vector<const DDsvalues_type*>& specs = fv.specifics();
  double nStrips = 0., nPads = 0., dPhi = 0.;
  for (auto const& is : specs) {
    if (DDfetch(is, numbOfStrips))
      nStrips = numbOfStrips.doubles()[0];
    if (DDfetch(is, numbOfPads))
      nPads = numbOfPads.doubles()[0];
    if (DDfetch(is, delPhi))
      dPhi = delPhi.doubles()[0];
  }
  LogDebug("GEMGeometryParsFromDD") << ((nStrips == 0.) ? ("No nStrips found!!")
                                                        : ("Number of strips: " + std::to_string(nStrips)));
  LogDebug("GEMGeometryParsFromDD") << ((nPads == 0.) ? ("No nPads found!!")
                                                      : ("Number of pads: " + std::to_string(nPads)));

  // EtaPartition specific parameter (size)
  std::vector<double> dpar = fv.logicalPart().solid().parameters();

  double dy = dpar[0];   //length is along local Y
  double dz = dpar[3];   //0.4;// thickness is long local Z
  double dx1 = dpar[4];  // bottom width is along local X
  double dx2 = dpar[8];  // top width is along local X

  std::vector<double> pars{dx1, dx2, dy, dz, nStrips, nPads, dPhi};
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  LogDebug("GEMGeometryParsFromDD") << " dx1 " << dx1 << " dx2 " << dx2 << " dy " << dy << " dz " << dz << " nStrips "
                                    << nStrips << " nPads " << nPads << " dPhi " << dPhi;

  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(5) DDD, Eta Partion DetID " << detId.rawId() << " Name " << fv.logicalPart().name().name() << " dx1 " << dx1
      << " dx2 " << dx2 << " dy " << dy << " dz " << dz << " nStrips " << nStrips << " nPads " << nPads << " dPhi "
      << dPhi;
  rgeo.insert(detId.rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

std::vector<double> GEMGeometryParsFromDD::getTranslation(DDFilteredView& fv) {
  const DDTranslation& tran = fv.translation();
  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(1) DDD, tran vector " << tran.x() << "  " << tran.y() << "  " << tran.z();
  return {tran.x(), tran.y(), tran.z()};
}

std::vector<double> GEMGeometryParsFromDD::getRotation(DDFilteredView& fv) {
  const DDRotationMatrix& rota = fv.rotation();  //.Inverse();
  DD3Vector x, y, z;
  rota.GetComponents(x, y, z);
  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(2) DDD, rot matrix " << x.X() << "  " << x.Y() << "  " << x.Z() << " " << y.X() << "  " << y.Y() << "  "
      << y.Z() << " " << z.X() << "  " << z.Y() << "  " << z.Z();
  return {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};
}

// DD4Hep

void GEMGeometryParsFromDD::build(const cms::DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rgeo) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapGEM";

  const cms::DDFilter filter(attribute, value);
  cms::DDFilteredView fv(*cview, filter);

  this->buildGeometry(fv, muonConstants, rgeo);
}

void GEMGeometryParsFromDD::buildGeometry(cms::DDFilteredView& fv,
                                          const MuonGeometryConstants& muonConstants,
                                          RecoIdealGeometry& rgeo) {
  edm::LogVerbatim("GEMGeometryParsFromDD") << "(0) GEMGeometryParsFromDD - DD4HEP ";

  MuonGeometryNumbering mdddnum(muonConstants);
  GEMNumberingScheme gemNum(muonConstants);
  static constexpr uint32_t levelChamb = 7;
  int chamb(0), region(0);
  int theLevelPart = muonConstants.getValue("level");
  int theRingLevel = muonConstants.getValue("mg_ring") / theLevelPart;
  int theSectorLevel = muonConstants.getValue("mg_sector") / theLevelPart;

  while (fv.firstChild()) {
    const auto& history = fv.history();
    MuonBaseNumber num(mdddnum.geoHistoryToBaseNumber(history));
    GEMDetId detId(gemNum.baseNumberToUnitNumber(num));

    if (detId.station() == GEMDetId::minStationId0) {
      if (num.getLevels() == theRingLevel) {
        if (detId.region() != region) {
          region = detId.region();
          chamb = 0;
        }
        ++chamb;
        detId = GEMDetId(detId.region(), detId.ring(), detId.station(), detId.layer(), chamb, 0);
        buildSuperChamber(fv, detId, rgeo);
      } else if (num.getLevels() == theSectorLevel) {
        buildChamber(fv, detId, rgeo);
      } else {
        buildEtaPartition(fv, detId, rgeo);
      }
    } else {
      if (fv.level() == levelChamb) {
        if (detId.layer() == 1) {
          buildSuperChamber(fv, detId, rgeo);
        }
        buildChamber(fv, detId, rgeo);
      } else if (num.getLevels() > theSectorLevel) {
        buildEtaPartition(fv, detId, rgeo);
      }
    }
  }
}

void GEMGeometryParsFromDD::buildSuperChamber(cms::DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo) {
  cms::DDSolid solid(fv.solid());
  auto solidA = solid.solidA();
  std::vector<double> dpar = solidA.dimensions();

  double dy = dpar[3] / dd4hep::mm;   //length is along local Y
  double dz = dpar[2] / dd4hep::mm;   // thickness is long local Z
  double dx1 = dpar[0] / dd4hep::mm;  // bottom width is along local X
  double dx2 = dpar[1] / dd4hep::mm;  // top width is along loc

  auto solidB = solid.solidB();
  dpar = solidB.dimensions();
  const int nch = 2;
  const double chgap = 2.105;

  GEMDetId gemid = detId.superChamberId();
  std::string_view name = fv.name();

  dz += (dpar[2] / dd4hep::mm);  // chamber thickness
  dz *= nch;                     // 2 chambers in superchamber
  dz += chgap;                   // gap between chambers

  std::vector<double> pars{dx1, dx2, dy, dz};
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(3) DD4HEP, SuperChamber DetID " << gemid.rawId() << " Name " << std::string(name) << " dx1 " << dx1
      << " dx2 " << dx2 << " dy " << dy << " dz " << dz;
  rgeo.insert(gemid.rawId(), vtra, vrot, pars, {std::string(name)});
}

void GEMGeometryParsFromDD::buildChamber(cms::DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo) {
  cms::DDSolid solid(fv.solid());
  auto solidA = solid.solidA();
  std::vector<double> dpar = solidA.dimensions();

  double dy = dpar[3] / dd4hep::mm;   //length is along local Y
  double dz = dpar[2] / dd4hep::mm;   // thickness is long local Z
  double dx1 = dpar[0] / dd4hep::mm;  // bottom width is along local X
  double dx2 = dpar[1] / dd4hep::mm;  // top width is along local X

  auto solidB = solid.solidB();
  dpar = solidB.dimensions();

  dz += (dpar[2] / dd4hep::mm);  // chamber thickness

  GEMDetId gemid = detId.chamberId();
  std::string_view name = fv.name();

  std::vector<double> pars{dx1, dx2, dy, dz};
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(4) DD4HEP, Chamber DetID " << gemid.rawId() << " Name " << std::string(name) << " dx1 " << dx1 << " dx2 "
      << dx2 << " dy " << dy << " dz " << dz;
  rgeo.insert(gemid.rawId(), vtra, vrot, pars, {std::string(name)});
}

void GEMGeometryParsFromDD::buildEtaPartition(cms::DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo) {
  auto nStrips = fv.get<double>("nStrips");
  auto nPads = fv.get<double>("nPads");
  auto dPhi = fv.get<double>("dPhi");

  std::vector<double> dpar = fv.parameters();
  std::string_view name = fv.name();

  double dx1 = dpar[0] / dd4hep::mm;
  double dx2 = dpar[1] / dd4hep::mm;
  double dy = dpar[3] / dd4hep::mm;
  double dz = dpar[2] / dd4hep::mm;

  std::vector<double> pars{dx1, dx2, dy, dz, nStrips, nPads, dPhi};
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(5) DD4HEP, Eta Partion DetID " << detId.rawId() << " Name " << std::string(name) << " dx1 " << dx1 << " dx2 "
      << dx2 << " dy " << dy << " dz " << dz << " nStrips " << nStrips << " nPads " << nPads << " dPhi " << dPhi;
  rgeo.insert(detId.rawId(), vtra, vrot, pars, {std::string(name)});
}

std::vector<double> GEMGeometryParsFromDD::getTranslation(cms::DDFilteredView& fv) {
  std::vector<double> tran(3);
  tran[0] = static_cast<double>(fv.translation().X()) / dd4hep::mm;
  tran[1] = static_cast<double>(fv.translation().Y()) / dd4hep::mm;
  tran[2] = static_cast<double>(fv.translation().Z()) / dd4hep::mm;

  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(1) DD4HEP, tran vector " << tran[0] << "  " << tran[1] << "  " << tran[2];
  return {tran[0], tran[1], tran[2]};
}

std::vector<double> GEMGeometryParsFromDD::getRotation(cms::DDFilteredView& fv) {
  DDRotationMatrix rota;
  fv.rot(rota);
  DD3Vector x, y, z;
  rota.GetComponents(x, y, z);
  const std::vector<double> rot = {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};
  edm::LogVerbatim("GEMGeometryParsFromDD")
      << "(2) DD4HEP, rot matrix " << rot[0] << "  " << rot[1] << "  " << rot[2] << " " << rot[3] << "  " << rot[4]
      << "  " << rot[5] << " " << rot[6] << "  " << rot[7] << "  " << rot[8];
  return {rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]};
}
