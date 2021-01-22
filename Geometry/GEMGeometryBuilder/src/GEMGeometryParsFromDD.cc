/** Implementation of the GEM Geometry Builder from DDD
 *
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include <iostream>
#include <algorithm>

GEMGeometryParsFromDD::GEMGeometryParsFromDD() {}

GEMGeometryParsFromDD::~GEMGeometryParsFromDD() {}

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

  double dy = dpar[0];   //length is along local Y
  double dz = dpar[3];   // thickness is long local Z
  double dx1 = dpar[4];  // bottom width is along local X
  double dx2 = dpar[8];  // top width is along local X
  dpar = solid.solidB().parameters();
  dz += dpar[3];  // chamber thickness
  dz *= 2;        // 2 chambers in superchamber
  dz += 2.105;    // gap between chambers

  GEMDetId gemid = detId.superChamberId();

  std::vector<double> pars{dx1, dx2, dy, dz};
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

  LogDebug("GEMGeometryParsFromDD") << "dimension dx1 " << dx1 << ", dx2 " << dx2 << ", dy " << dy << ", dz " << dz;
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

  LogDebug("GEMGeometryParsFromDD") << "dimension dx1 " << dx1 << ", dx2 " << dx2 << ", dy " << dy << ", dz " << dz;
  rgeo.insert(detId.rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

std::vector<double> GEMGeometryParsFromDD::getTranslation(DDFilteredView& fv) {
  const DDTranslation& tran = fv.translation();
  return {tran.x(), tran.y(), tran.z()};
}

std::vector<double> GEMGeometryParsFromDD::getRotation(DDFilteredView& fv) {
  const DDRotationMatrix& rota = fv.rotation();  //.Inverse();
  DD3Vector x, y, z;
  rota.GetComponents(x, y, z);
  return {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};
}
