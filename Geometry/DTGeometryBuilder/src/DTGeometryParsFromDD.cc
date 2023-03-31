/** \class DTGeometryParsFromDD
 *
 *  Build the RPCGeometry from the DDD and DD4hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by Stefano Lacaprara (INFN LNL)
 *  \author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Tue, 26 Jan 2021 
 *
 */
#include <Geometry/DTGeometryBuilder/interface/DTGeometryParsFromDD.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>
#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include <DetectorDescription/DDCMS/interface/DDFilteredView.h>
#include <DetectorDescription/DDCMS/interface/DDCompactView.h>
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <algorithm>
#include <string>

using namespace std;
using namespace geant_units;
using namespace geant_units::operators;

DTGeometryParsFromDD::DTGeometryParsFromDD() {}

DTGeometryParsFromDD::~DTGeometryParsFromDD() {}

// DD
void DTGeometryParsFromDD::build(const DDCompactView* cview,
                                 const MuonGeometryConstants& muonConstants,
                                 RecoIdealGeometry& rig) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "DTGeometryParsFromDD::build";
#endif

  std::string attribute = "MuStructure";
  std::string value = "MuonBarrelDT";

  // Asking only for the Muon DTs
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview, filter);
  buildGeometry(fview, muonConstants, rig);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "RecoIdealGeometry " << rig.size();
#endif
}

// DD4hep

void DTGeometryParsFromDD::build(const cms::DDCompactView* cview,
                                 const MuonGeometryConstants& muonConstants,
                                 RecoIdealGeometry& rgeo) {
  const std::string attribute = "MuStructure";
  const std::string value = "MuonBarrelDT";
  const cms::DDFilter filter(attribute, value);
  cms::DDFilteredView fview(*cview, filter);
  buildGeometry(fview, muonConstants, rgeo);
}

// DD
void DTGeometryParsFromDD::buildGeometry(DDFilteredView& fv,
                                         const MuonGeometryConstants& muonConstants,
                                         RecoIdealGeometry& rig) const {
  // static const string t0 = "DTGeometryParsFromDD::buildGeometry";
  // TimeMe timer(t0,true);

  edm::LogVerbatim("DTGeometryParsFromDD") << "(0) DTGeometryParsFromDD - DDD ";

  bool doChamber = fv.firstChild();

  // Loop on chambers
  int ChamCounter = 0;
  while (doChamber) {
    ChamCounter++;
    DDValue val("Type");
    const DDsvalues_type params(fv.mergedSpecifics());
    string type;
    if (DDfetch(&params, val))
      type = val.strings()[0];
    // FIXME
    val = DDValue("FEPos");
    string FEPos;
    if (DDfetch(&params, val))
      FEPos = val.strings()[0];
    insertChamber(fv, type, muonConstants, rig);

    // Loop on SLs
    bool doSL = fv.firstChild();
    int SLCounter = 0;
    while (doSL) {
      SLCounter++;
      insertSuperLayer(fv, type, muonConstants, rig);

      bool doL = fv.firstChild();
      int LCounter = 0;
      // Loop on SLs
      while (doL) {
        LCounter++;
        insertLayer(fv, type, muonConstants, rig);

        // fv.parent();
        doL = fv.nextSibling();  // go to next layer
      }                          // layers

      fv.parent();
      doSL = fv.nextSibling();  // go to next SL
    }                           // sls

    fv.parent();
    doChamber = fv.nextSibling();  // go to next chamber
  }                                // chambers
}

void DTGeometryParsFromDD::insertChamber(DDFilteredView& fv,
                                         const string& type,
                                         const MuonGeometryConstants& muonConstants,
                                         RecoIdealGeometry& rig) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTChamberId detId(rawid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "inserting Chamber " << detId;
#endif

  // Chamber specific parameter (size)
  vector<double> par;
  par.emplace_back(DTChamberTag);
  vector<double> size = extractParameters(fv);
  par.insert(par.end(), size.begin(), size.end());
  edm::LogVerbatim("DTGeometryParsFromDD")
      << "(1) DDD, Chamber DetID " << rawid << " " << par[0] << " " << par[1] << " " << par[2] << " " << par[3];
  ///SL the definition of length, width, thickness depends on the local reference frame of the Det
  // width is along local X
  // length is along local Y
  // thickness is long local Z

  PosRotPair posRot(plane(fv));
  rig.insert(rawid, posRot.first, posRot.second, par);
}

void DTGeometryParsFromDD::insertSuperLayer(DDFilteredView& fv,
                                            const std::string& type,
                                            const MuonGeometryConstants& muonConstants,
                                            RecoIdealGeometry& rig) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTSuperLayerId slId(rawid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "inserting SuperLayer " << slId;
#endif

  // Slayer specific parameter (size)
  vector<double> par;
  par.emplace_back(DTSuperLayerTag);
  vector<double> size = extractParameters(fv);
  par.insert(par.end(), size.begin(), size.end());
  edm::LogVerbatim("DTGeometryParsFromDD")
      << "(2) DDD, Super Layer DetID " << rawid << " " << par[0] << " " << par[1] << " " << par[2] << " " << par[3];
  // Ok this is the slayer position...
  PosRotPair posRot(plane(fv));
  rig.insert(slId, posRot.first, posRot.second, par);
}

void DTGeometryParsFromDD::insertLayer(DDFilteredView& fv,
                                       const std::string& type,
                                       const MuonGeometryConstants& muonConstants,
                                       RecoIdealGeometry& rig) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTLayerId layId(rawid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "inserting Layer " << layId;
#endif
  // Layer specific parameter (size)
  vector<double> par;
  par.emplace_back(DTLayerTag);
  vector<double> size = extractParameters(fv);
  par.insert(par.end(), size.begin(), size.end());

  // Loop on wires
  bool doWire = fv.firstChild();
  int WCounter = 0;
  int firstWire = fv.copyno();
  //float wireLength = par[1]/cm;
  while (doWire) {
    WCounter++;
    doWire = fv.nextSibling();  // next wire
  }
  vector<double> sensSize = extractParameters(fv);
  //int lastWire=fv.copyno();
  par.emplace_back(firstWire);
  par.emplace_back(WCounter);
  par.emplace_back(sensSize[1]);
  fv.parent();

  PosRotPair posRot(plane(fv));

  edm::LogVerbatim("DTGeometryParsFromDD")
      << "(3) DDD, Layer DetID " << rawid << " " << par[0] << " " << par[1] << " " << par[2] << " " << par[3] << " "
      << par[4] << " " << par[5] << " " << par[6];
  rig.insert(layId, posRot.first, posRot.second, par);
}

vector<double> DTGeometryParsFromDD::extractParameters(DDFilteredView& fv) const {
  vector<double> par;
  if (fv.logicalPart().solid().shape() != DDSolidShape::ddbox) {
    DDBooleanSolid bs(fv.logicalPart().solid());
    DDSolid A = bs.solidA();
    while (A.shape() != DDSolidShape::ddbox) {
      DDBooleanSolid bs(A);
      A = bs.solidA();
    }
    par = A.parameters();
  } else {
    par = fv.logicalPart().solid().parameters();
  }
  return par;
}

DTGeometryParsFromDD::PosRotPair DTGeometryParsFromDD::plane(const DDFilteredView& fv) const {
  // extract the position
  const DDTranslation& trans(fv.translation());

  std::vector<double> gtran(3);
  gtran[0] = convertMmToCm(trans.x());
  gtran[1] = convertMmToCm(trans.y());
  gtran[2] = convertMmToCm(trans.z());

  edm::LogVerbatim("DTGeometryParsFromDD") << "(4) DDD, Position "
                                           << " " << gtran[0] << " " << gtran[1] << " " << gtran[2];
  // now the rotation
  //     'active' and 'passive' rotations are inverse to each other
  const DDRotationMatrix& rotation = fv.rotation();  //REMOVED .Inverse();
  DD3Vector x, y, z;
  rotation.GetComponents(x, y, z);

  std::vector<double> grmat(9);
  grmat[0] = x.X();
  grmat[1] = x.Y();
  grmat[2] = x.Z();

  grmat[3] = y.X();
  grmat[4] = y.Y();
  grmat[5] = y.Z();

  grmat[6] = z.X();
  grmat[7] = z.Y();
  grmat[8] = z.Z();

  return pair<std::vector<double>, std::vector<double> >(gtran, grmat);
}

// DD4hep

void DTGeometryParsFromDD::buildGeometry(cms::DDFilteredView& fv,
                                         const MuonGeometryConstants& muonConstants,
                                         RecoIdealGeometry& rig) const {
  edm::LogVerbatim("DTGeometryParsFromDD") << "(0) DTGeometryParsFromDD - DD4hep ";

  bool doChamber = fv.firstChild();

  while (doChamber) {
    insertChamber(fv, muonConstants, rig);

    bool doSL = fv.nextSibling();
    while (doSL) {
      insertSuperLayer(fv, muonConstants, rig);

      fv.down();
      bool doLayers = fv.sibling();
      while (doLayers) {
        insertLayer(fv, muonConstants, rig);

        doLayers = fv.sibling();
      }

      doSL = fv.nextSibling();
    }

    fv.parent();
    doChamber = fv.firstChild();
  }
}

DTGeometryParsFromDD::PosRotPair DTGeometryParsFromDD::plane(const cms::DDFilteredView& fv) const {
  const Double_t* tr = fv.trans();

  std::vector<double> gtran(3);

  gtran[0] = tr[0] / dd4hep::cm;
  gtran[1] = tr[1] / dd4hep::cm;
  gtran[2] = tr[2] / dd4hep::cm;

  edm::LogVerbatim("DTGeometryParsFromDD") << "(4) DD4hep, Position "
                                           << " " << gtran[0] << " " << gtran[1] << " " << gtran[2];

  DDRotationMatrix rotation = fv.rotation();
  DD3Vector x, y, z;
  rotation.GetComponents(x, y, z);

  std::vector<double> grmat(9);

  grmat[0] = x.X();
  grmat[1] = x.Y();
  grmat[2] = x.Z();

  grmat[3] = y.X();
  grmat[4] = y.Y();
  grmat[5] = y.Z();

  grmat[6] = z.X();
  grmat[7] = z.Y();
  grmat[8] = z.Z();

  return pair<std::vector<double>, std::vector<double> >(gtran, grmat);
}

void DTGeometryParsFromDD::insertChamber(cms::DDFilteredView& fv,
                                         const MuonGeometryConstants& muonConstants,
                                         RecoIdealGeometry& rig) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.history()));
  DTChamberId detId(rawid);

  vector<double> par_temp = fv.parameters();
  vector<double> par(4);
  par[0] = DTChamberTag;  //DTChamberTag is the ID of a Chamber
  par[1] = par_temp[0] / dd4hep::mm;
  par[2] = par_temp[1] / dd4hep::mm;
  par[3] = par_temp[2] / dd4hep::mm;

  edm::LogVerbatim("DTGeometryParsFromDD")
      << "(1) DD4hep, Chamber DetID " << rawid << " " << par[0] << " " << par[1] << " " << par[2] << " " << par[3];
  PosRotPair posRot(plane(fv));
  rig.insert(rawid, posRot.first, posRot.second, par);
}

void DTGeometryParsFromDD::insertSuperLayer(cms::DDFilteredView& fv,
                                            const MuonGeometryConstants& muonConstants,
                                            RecoIdealGeometry& rig) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.history()));
  DTSuperLayerId slId(rawid);

  vector<double> par_temp = fv.parameters();
  vector<double> par(4);
  par[0] = DTSuperLayerTag;  //DTSuperLayerTag is the ID of a SuperLayer
  par[1] = par_temp[0] / dd4hep::mm;
  par[2] = par_temp[1] / dd4hep::mm;
  par[3] = par_temp[2] / dd4hep::mm;

  edm::LogVerbatim("DTGeometryParsFromDD")
      << "(2) DD4hep, Super Layer DetID " << rawid << " " << par[0] << " " << par[1] << " " << par[2] << " " << par[3];
  PosRotPair posRot(plane(fv));
  rig.insert(slId, posRot.first, posRot.second, par);
}

void DTGeometryParsFromDD::insertLayer(cms::DDFilteredView& fv,
                                       const MuonGeometryConstants& muonConstants,
                                       RecoIdealGeometry& rig) const {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.history()));
  DTLayerId layId(rawid);

  vector<double> par_temp = fv.parameters();
  vector<double> par(4);
  par[0] = DTLayerTag;  //DTLayerTag is the ID of a Layer
  par[1] = par_temp[0] / dd4hep::mm;
  par[2] = par_temp[1] / dd4hep::mm;
  par[3] = par_temp[2] / dd4hep::mm;

  fv.down();
  bool doWire = fv.sibling();
  int firstWire = fv.volume()->GetNumber();
  auto const& wpar = fv.parameters();
  float wireLength = wpar[1] / dd4hep::mm;

  int WCounter = 0;
  while (doWire) {
    doWire = fv.checkChild();
    WCounter++;
  }

  par.emplace_back(firstWire);
  par.emplace_back(WCounter);
  par.emplace_back(wireLength);

  fv.up();

  PosRotPair posRot(plane(fv));

  edm::LogVerbatim("DTGeometryParsFromDD")
      << "(3) DD4hep, Layer DetID " << rawid << " " << par[0] << " " << par[1] << " " << par[2] << " " << par[3] << " "
      << par[4] << " " << par[5] << " " << par[6];
  rig.insert(layId, posRot.first, posRot.second, par);
}
