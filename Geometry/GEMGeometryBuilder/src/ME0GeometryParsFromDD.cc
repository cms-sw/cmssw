/* Implementation of the  ME0GeometryParsFromDD Class
 *  Build the ME0Geometry from the DDD and DD4Hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
 *  Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Thu, 25 Feb 2021 
 *
 */
#include "Geometry/GEMGeometryBuilder/interface/ME0GeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include <iostream>
#include <algorithm>

// DD

void ME0GeometryParsFromDD::build(const DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rgeo) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapME0";

  // Asking only for the MuonME0's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview, filter);

  this->buildGeometry(fview, muonConstants, rgeo);
}

void ME0GeometryParsFromDD::buildGeometry(DDFilteredView& fv,
                                          const MuonGeometryConstants& muonConstants,
                                          RecoIdealGeometry& rgeo) {
  LogDebug("ME0GeometryParsFromDD") << "Building the geometry service";
  LogDebug("ME0GeometryParsFromDD") << "About to run through the ME0 structure\n"
                                    << " First logical part " << fv.logicalPart().name().name();

  edm::LogVerbatim("ME0GeometryParsFromDD") << "(0) ME0GeometryParsFromDD - DDD ";
  MuonGeometryNumbering muonDDDNumbering(muonConstants);
  ME0NumberingScheme me0Numbering(muonConstants);

  bool doChambers = fv.firstChild();
  LogDebug("ME0GeometryParsFromDD") << "doChamber = " << doChambers;
  // loop over superchambers
  while (doChambers) {
    // getting chamber id from eta partitions
    fv.firstChild();
    fv.firstChild();
    ME0DetId detIdCh =
        ME0DetId(me0Numbering.baseNumberToUnitNumber(muonDDDNumbering.geoHistoryToBaseNumber(fv.geoHistory())));
    // back to chambers
    fv.parent();
    fv.parent();

    buildChamber(fv, detIdCh, rgeo);

    // loop over chambers
    // only 1 chamber
    bool doLayers = fv.firstChild();
    while (doLayers) {
      // get layer ID
      fv.firstChild();
      ME0DetId detIdLa =
          ME0DetId(me0Numbering.baseNumberToUnitNumber(muonDDDNumbering.geoHistoryToBaseNumber(fv.geoHistory())));
      fv.parent();
      // build layer
      buildLayer(fv, detIdLa, rgeo);

      // loop over ME0EtaPartitions
      bool doEtaPart = fv.firstChild();
      while (doEtaPart) {
        ME0DetId detId =
            ME0DetId(me0Numbering.baseNumberToUnitNumber(muonDDDNumbering.geoHistoryToBaseNumber(fv.geoHistory())));
        buildEtaPartition(fv, detId, rgeo);

        doEtaPart = fv.nextSibling();
      }
      fv.parent();
      doLayers = fv.nextSibling();
    }
    fv.parent();
    doChambers = fv.nextSibling();
  }
}

void ME0GeometryParsFromDD::buildChamber(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo) {
  LogDebug("ME0GeometryParsFromDD") << "buildChamber " << fv.logicalPart().name().name() << " " << detId << std::endl;

  std::vector<double> pars = getDimension(fv);
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);
  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(4) DDD, Chamber DetID " << detId.chamberId().rawId() << " Name " << fv.logicalPart().name().name();

  rgeo.insert(detId.chamberId().rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

void ME0GeometryParsFromDD::buildLayer(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo) {
  LogDebug("ME0GeometryParsFromDD") << "buildLayer " << fv.logicalPart().name().name() << " " << detId << std::endl;

  std::vector<double> pars = getDimension(fv);
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(5) DDD, Layer DetID " << detId.layerId().rawId() << " Name " << fv.logicalPart().name().name();
  rgeo.insert(detId.layerId().rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

void ME0GeometryParsFromDD::buildEtaPartition(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo) {
  LogDebug("ME0GeometryParsFromDD") << "buildEtaPartition " << fv.logicalPart().name().name() << " " << detId
                                    << std::endl;

  // EtaPartition specific parameter (nstrips and npads)
  DDValue numbOfStrips("nStrips");
  DDValue numbOfPads("nPads");
  const std::vector<const DDsvalues_type*>& specs = fv.specifics();
  double nStrips = 0., nPads = 0.;
  for (auto const& is : specs) {
    if (DDfetch(is, numbOfStrips))
      nStrips = numbOfStrips.doubles()[0];
    if (DDfetch(is, numbOfPads))
      nPads = numbOfPads.doubles()[0];
  }
  LogDebug("ME0GeometryParsFromDD") << ((nStrips == 0.) ? ("No nStrips found!!")
                                                        : ("Number of strips: " + std::to_string(nStrips)));
  LogDebug("ME0GeometryParsFromDD") << ((nPads == 0.) ? ("No nPads found!!")
                                                      : ("Number of pads: " + std::to_string(nPads)));

  std::vector<double> pars = getDimension(fv);
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(6) DDD, Eta Partion DetID " << detId.rawId() << " Name " << fv.logicalPart().name().name() << " nStrips "
      << nStrips << " nPads " << nPads;

  rgeo.insert(detId.rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

std::vector<double> ME0GeometryParsFromDD::getDimension(DDFilteredView& fv) {
  std::vector<double> dpar = fv.logicalPart().solid().parameters();
  //dpar[4] bottom width is along local X
  //dpar[8] top width is along local X
  //dpar[3] thickness is long local Z
  //dpar[0] length is along local Y
  LogDebug("ME0GeometryParsFromDD") << "dimension dx1 " << dpar[4] << ", dx2 " << dpar[8] << ", dy " << dpar[0]
                                    << ", dz " << dpar[3];
  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(1) DDD, dimension dx1 " << dpar[4] << ", dx2 " << dpar[8] << ", dy " << dpar[0] << ", dz " << dpar[3];
  return {dpar[4], dpar[8], dpar[0], dpar[3]};
}

std::vector<double> ME0GeometryParsFromDD::getTranslation(DDFilteredView& fv) {
  const DDTranslation& tran = fv.translation();
  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(2) DDD, tran vector " << tran.x() << "  " << tran.y() << "  " << tran.z();
  return {tran.x(), tran.y(), tran.z()};
}

std::vector<double> ME0GeometryParsFromDD::getRotation(DDFilteredView& fv) {
  const DDRotationMatrix& rota = fv.rotation();  //.Inverse();
  DD3Vector x, y, z;
  rota.GetComponents(x, y, z);
  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(3) DDD, rot matrix " << x.X() << "  " << x.Y() << "  " << x.Z() << " " << y.X() << "  " << y.Y() << "  "
      << y.Z() << " " << z.X() << "  " << z.Y() << "  " << z.Z();
  return {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};
}

// DD4HEP

void ME0GeometryParsFromDD::build(const cms::DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rgeo) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapME0";
  const cms::DDFilter filter(attribute, value);
  cms::DDFilteredView fview(*cview, filter);
  this->buildGeometry(fview, muonConstants, rgeo);
}

void ME0GeometryParsFromDD::buildGeometry(cms::DDFilteredView& fv,
                                          const MuonGeometryConstants& muonConstants,
                                          RecoIdealGeometry& rgeo) {
  edm::LogVerbatim("ME0GeometryParsFromDD") << "(0) ME0GeometryParsFromDD - DD4HEP ";

  MuonGeometryNumbering mdddnum(muonConstants);
  ME0NumberingScheme me0Num(muonConstants);

  static constexpr uint32_t levelChamber = 7;
  static constexpr uint32_t levelLayer = 8;
  uint32_t theLevelPart = muonConstants.getValue("level");
  uint32_t theSectorLevel = muonConstants.getValue("m0_sector") / theLevelPart;

  while (fv.firstChild()) {
    const auto& history = fv.history();
    MuonBaseNumber num(mdddnum.geoHistoryToBaseNumber(history));
    ME0DetId detId(me0Num.baseNumberToUnitNumber(num));

    if (fv.level() == levelChamber) {
      buildChamber(fv, detId, rgeo);
    } else if (fv.level() == levelLayer) {
      buildLayer(fv, detId, rgeo);
    } else if (history.tags.size() > theSectorLevel) {
      buildEtaPartition(fv, detId, rgeo);
    }
  }  // end while
}  // end buildGeometry

void ME0GeometryParsFromDD::buildChamber(cms::DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo) {
  std::string_view name = fv.name();
  std::vector<double> pars = getDimension(fv);
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(4) DD4HEP, Chamber DetID " << detId.chamberId().rawId() << " Name " << std::string(name);

  rgeo.insert(detId.chamberId().rawId(), vtra, vrot, pars, {std::string(name)});
}

void ME0GeometryParsFromDD::buildLayer(cms::DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo) {
  std::string_view name = fv.name();
  std::vector<double> pars = getDimension(fv);
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(5) DD4HEP, Layer DetID " << detId.layerId().rawId() << " Name " << std::string(name);
  rgeo.insert(detId.layerId().rawId(), vtra, vrot, pars, {std::string(name)});
}

void ME0GeometryParsFromDD::buildEtaPartition(cms::DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo) {
  auto nStrips = fv.get<double>("nStrips");
  auto nPads = fv.get<double>("nPads");
  std::string_view name = fv.name();
  std::vector<double> pars = getDimension(fv);
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  edm::LogVerbatim("ME0GeometryParsFromDD") << "(6) DD4HEP, Eta Partion DetID " << detId.rawId() << " Name "
                                            << std::string(name) << " nStrips " << nStrips << " nPads " << nPads;

  rgeo.insert(detId.rawId(), vtra, vrot, pars, {std::string(name)});
}

std::vector<double> ME0GeometryParsFromDD::getDimension(cms::DDFilteredView& fv) {
  std::vector<double> dpar = fv.parameters();

  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(1) DD4HEP, dimension dx1 " << dpar[0] / dd4hep::mm << ", dx2 " << dpar[1] / dd4hep::mm << ", dy "
      << dpar[3] / dd4hep::mm << ", dz " << dpar[2] / dd4hep::mm;

  return {dpar[0] / dd4hep::mm, dpar[1] / dd4hep::mm, dpar[3] / dd4hep::mm, dpar[2] / dd4hep::mm};
}

std::vector<double> ME0GeometryParsFromDD::getTranslation(cms::DDFilteredView& fv) {
  std::vector<double> tran(3);
  tran[0] = static_cast<double>(fv.translation().X()) / dd4hep::mm;
  tran[1] = static_cast<double>(fv.translation().Y()) / dd4hep::mm;
  tran[2] = static_cast<double>(fv.translation().Z()) / dd4hep::mm;

  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(2) DD4HEP, tran vector " << tran[0] << "  " << tran[1] << "  " << tran[2];
  return {tran[0], tran[1], tran[2]};
}

std::vector<double> ME0GeometryParsFromDD::getRotation(cms::DDFilteredView& fv) {
  DDRotationMatrix rota;
  fv.rot(rota);
  DD3Vector x, y, z;
  rota.GetComponents(x, y, z);
  const std::vector<double> rot = {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};
  edm::LogVerbatim("ME0GeometryParsFromDD")
      << "(3) DD4HEP, rot matrix " << rot[0] << "  " << rot[1] << "  " << rot[2] << " " << rot[3] << "  " << rot[4]
      << "  " << rot[5] << " " << rot[6] << "  " << rot[7] << "  " << rot[8];
  return {rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]};
}
