#include "Geometry/GEMGeometryBuilder/src/ME0GeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

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

  rgeo.insert(detId.chamberId().rawId(), vtra, vrot, pars, {fv.logicalPart().name().name()});
}

void ME0GeometryParsFromDD::buildLayer(DDFilteredView& fv, ME0DetId detId, RecoIdealGeometry& rgeo) {
  LogDebug("ME0GeometryParsFromDD") << "buildLayer " << fv.logicalPart().name().name() << " " << detId << std::endl;

  std::vector<double> pars = getDimension(fv);
  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);

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
  return {dpar[4], dpar[8], dpar[0], dpar[3]};
}

std::vector<double> ME0GeometryParsFromDD::getTranslation(DDFilteredView& fv) {
  const DDTranslation& tran = fv.translation();
  return {tran.x(), tran.y(), tran.z()};
}

std::vector<double> ME0GeometryParsFromDD::getRotation(DDFilteredView& fv) {
  const DDRotationMatrix& rota = fv.rotation();  //.Inverse();
  DD3Vector x, y, z;
  rota.GetComponents(x, y, z);
  return {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};
}
