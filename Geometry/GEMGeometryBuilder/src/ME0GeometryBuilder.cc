/*
//\class ME0GeometryBuilder

 Description: ME0 Geometry builder from DD & DD4hep
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  29 Apr 2019 
*/
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilder.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include <algorithm>
#include <iostream>
#include <string>

using namespace geant_units::operators;

ME0GeometryBuilder::ME0GeometryBuilder() {}

ME0GeometryBuilder::~ME0GeometryBuilder() {}

ME0Geometry* ME0GeometryBuilder::build(const DDCompactView* cview, const MuonGeometryConstants& muonConstants) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapME0";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview, filter);
  return this->buildGeometry(fview, muonConstants);
}

// for DD4hep
ME0Geometry* ME0GeometryBuilder::build(const cms::DDCompactView* cview, const MuonGeometryConstants& muonConstants) {
  std::string attribute = "MuStructure";
  std::string value = "MuonEndCapME0";
  cms::DDFilteredView fview(cview->detector(), cview->detector()->worldVolume());
  cms::DDSpecParRefs refs;
  const cms::DDSpecParRegistry& mypar = cview->specpars();
  mypar.filter(refs, attribute, value);
  fview.mergedSpecifics(refs);
  return this->buildGeometry(fview, muonConstants);
}

ME0Geometry* ME0GeometryBuilder::buildGeometry(DDFilteredView& fv, const MuonGeometryConstants& muonConstants) {
  ME0Geometry* geometry = new ME0Geometry();
  MuonGeometryNumbering mdddnum(muonConstants);
  ME0NumberingScheme me0Num(muonConstants);

  LogTrace("ME0GeometryBuilder") << "Building the geometry service";
  LogTrace("ME0GeometryBuilder") << "About to run through the ME0 structure\n"
                                 << "Top level logical part: " << fv.logicalPart().name().name();

// ==========================================
// ===  Test to understand the structure  ===
// ==========================================
#ifdef EDM_ML_DEBUG
  bool testChambers = fv.firstChild();
  LogTrace("ME0GeometryBuilder") << "doChamber = fv.firstChild() = " << testChambers;

  while (testChambers) {
    // to etapartitions
    LogTrace("ME0GeometryBuilder") << "to layer " << fv.firstChild();
    LogTrace("ME0GeometryBuilder") << "to etapt " << fv.firstChild();
    int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
    ME0DetId detId = ME0DetId(rawId);
    ME0DetId detIdCh = detId.chamberId();
    // back to chambers
    LogTrace("ME0GeometryBuilder") << "back to layer " << fv.parent();
    LogTrace("ME0GeometryBuilder") << "back to chamb " << fv.parent();

    LogTrace("ME0GeometryBuilder") << "In DoChambers Loop :: ME0DetId " << detId << " = " << detId.rawId()
                                   << " (which belongs to ME0Chamber " << detIdCh << " = " << detIdCh.rawId() << ")";
    LogTrace("ME0GeometryBuilder") << "Second level logical part: " << fv.logicalPart().name().name();
    DDBooleanSolid solid2 = (DDBooleanSolid)(fv.logicalPart().solid());
    std::vector<double> dpar2 = solid2.parameters();
    std::stringstream parameters2;
    for (unsigned int i = 0; i < dpar2.size(); ++i) {
      parameters2 << " dpar[" << i << "]=" << convertMmToCm(dpar2[i]) << "cm ";
    }
    LogTrace("ME0GeometryBuilder") << "Second level parameters: vector with size = " << dpar2.size() << " and elements "
                                   << parameters2.str();

    bool doLayers = fv.firstChild();

    LogTrace("ME0GeometryBuilder") << "doLayer = fv.firstChild() = " << doLayers;
    while (doLayers) {
      // to etapartitions
      LogTrace("ME0GeometryBuilder") << "to etapt " << fv.firstChild();
      int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
      ME0DetId detId = ME0DetId(rawId);
      ME0DetId detIdLa = detId.layerId();
      // back to layers
      LogTrace("ME0GeometryBuilder") << "back to layer " << fv.parent();
      LogTrace("ME0GeometryBuilder") << "In DoLayers Loop :: ME0DetId " << detId << " = " << detId.rawId()
                                     << " (which belongs to ME0Layer " << detIdLa << " = " << detIdLa.rawId() << ")";
      LogTrace("ME0GeometryBuilder") << "Third level logical part: " << fv.logicalPart().name().name();
      DDBooleanSolid solid3 = (DDBooleanSolid)(fv.logicalPart().solid());
      std::vector<double> dpar3 = solid3.parameters();
      std::stringstream parameters3;
      for (unsigned int i = 0; i < dpar3.size(); ++i) {
        parameters3 << " dpar[" << i << "]=" << convertMmToCm(dpar3[i]) << "cm ";
      }
      LogTrace("ME0GeometryBuilder") << "Third level parameters: vector with size = " << dpar3.size()
                                     << " and elements " << parameters3.str();
      bool doEtaParts = fv.firstChild();

      LogTrace("ME0GeometryBuilder") << "doEtaPart = fv.firstChild() = " << doEtaParts;
      while (doEtaParts) {
        LogTrace("ME0GeometryBuilder") << "In DoEtaParts Loop :: ME0DetId " << detId << " = " << detId.rawId();
        LogTrace("ME0GeometryBuilder") << "Fourth level logical part: " << fv.logicalPart().name().name();
        DDBooleanSolid solid4 = (DDBooleanSolid)(fv.logicalPart().solid());
        std::vector<double> dpar4 = solid4.parameters();
        std::stringstream parameters4;
        for (unsigned int i = 0; i < dpar4.size(); ++i) {
          parameters4 << " dpar[" << i << "]=" << convertMmToCm(dpar4[i]) << "cm ";
        }
        LogTrace("ME0GeometryBuilder") << "Fourth level parameters: vector with size = " << dpar4.size()
                                       << " and elements " << parameters4.str();

        doEtaParts = fv.nextSibling();
        LogTrace("ME0GeometryBuilder") << "doEtaPart = fv.nextSibling() = " << doEtaParts;
      }
      fv.parent();
      LogTrace("ME0GeometryBuilder") << "went back to parent :: name = " << fv.logicalPart().name().name()
                                     << " will now ask for nextSibling";
      doLayers = fv.nextSibling();
      LogTrace("ME0GeometryBuilder") << "doLayer = fv.nextSibling() = " << doLayers;
    }
    fv.parent();
    LogTrace("ME0GeometryBuilder") << "went back to parent :: name = " << fv.logicalPart().name().name()
                                   << " will now ask for nextSibling";
    testChambers = fv.nextSibling();
    LogTrace("ME0GeometryBuilder") << "doChamber = fv.nextSibling() = " << testChambers;
  }
  fv.parent();
#endif

  // ==========================================
  // === Here the Real ME0 Geometry Builder ===
  // ==========================================
  bool doChambers = fv.firstChild();

  while (doChambers) {
    // to etapartitions and back again to pick up DetId
    fv.firstChild();
    fv.firstChild();

    int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
    ME0DetId detId = ME0DetId(rawId);
    ME0DetId detIdCh = detId.chamberId();

    fv.parent();
    fv.parent();

    // build chamber
    ME0Chamber* me0Chamber = buildChamber(fv, detIdCh);
    geometry->add(me0Chamber);

    // loop over layers of the chamber
    bool doLayers = fv.firstChild();

    while (doLayers) {
      // to etapartitions and back again to pick up DetId
      fv.firstChild();
      int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
      ME0DetId detId = ME0DetId(rawId);
      ME0DetId detIdLa = detId.layerId();
      fv.parent();
      // build layer
      ME0Layer* me0Layer = buildLayer(fv, detIdLa);
      me0Chamber->add(me0Layer);
      geometry->add(me0Layer);

      // loop over etapartitions of the layer
      bool doEtaParts = fv.firstChild();

      while (doEtaParts) {
        // pick up DetId
        int rawId = me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
        ME0DetId detId = ME0DetId(rawId);

        // build etapartition
        ME0EtaPartition* etaPart = buildEtaPartition(fv, detId);
        me0Layer->add(etaPart);
        geometry->add(etaPart);

        doEtaParts = fv.nextSibling();
      }
      fv.parent();

      doLayers = fv.nextSibling();
    }
    fv.parent();

    doChambers = fv.nextSibling();
  }
  return geometry;
}

ME0Chamber* ME0GeometryBuilder::buildChamber(DDFilteredView& fv, ME0DetId detId) const {
  LogTrace("ME0GeometryBuilder") << "buildChamber " << fv.logicalPart().name().name() << " " << detId << std::endl;
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());

  std::vector<double> dpar = solid.parameters();

  double L = convertMmToCm(dpar[0]);  // length is along local Y
  double T = convertMmToCm(dpar[3]);  // thickness is long local Z
  double b = convertMmToCm(dpar[4]);  // bottom width is along local X
  double B = convertMmToCm(dpar[8]);  // top width is along local X

#ifdef EDM_ML_DEBUG
  LogTrace("ME0GeometryBuilder") << " name of logical part = " << fv.logicalPart().name().name() << std::endl;
  LogTrace("ME0GeometryBuilder") << " dpar is vector with size = " << dpar.size() << std::endl;
  for (unsigned int i = 0; i < dpar.size(); ++i) {
    LogTrace("ME0GeometryBuilder") << " dpar [" << i << "] = " << convertMmToCm(dpar[i]) << " cm " << std::endl;
  }
  LogTrace("ME0GeometryBuilder") << "size  b: " << b << "cm, B: " << B << "cm,  L: " << L << "cm, T: " << T << "cm "
                                 << std::endl;
#endif

  bool isOdd = false;  // detId.chamber()%2;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b, B, L, T), isOdd));
  ME0Chamber* chamber = new ME0Chamber(detId.chamberId(), surf);
  return chamber;
}

ME0Layer* ME0GeometryBuilder::buildLayer(DDFilteredView& fv, ME0DetId detId) const {
  LogTrace("ME0GeometryBuilder") << "buildLayer " << fv.logicalPart().name().name() << " " << detId << std::endl;

  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());

  std::vector<double> dpar = solid.parameters();
  double L = convertMmToCm(dpar[0]);  // length is along local Y
  double t = convertMmToCm(dpar[3]);  // thickness is long local Z
  double b = convertMmToCm(dpar[4]);  // bottom width is along local X
  double B = convertMmToCm(dpar[8]);  // top width is along local X

#ifdef EDM_ML_DEBUG
  LogTrace("ME0GeometryBuilder") << " name of logical part = " << fv.logicalPart().name().name() << std::endl;
  LogTrace("ME0GeometryBuilder") << " dpar is vector with size = " << dpar.size() << std::endl;
  for (unsigned int i = 0; i < dpar.size(); ++i) {
    LogTrace("ME0GeometryBuilder") << " dpar [" << i << "] = " << convertMmToCm(dpar[i]) << " cm " << std::endl;
  }
  LogTrace("ME0GeometryBuilder") << "size  b: " << b << "cm, B: " << B << "cm,  L: " << L << "cm, t: " << t << "cm "
                                 << std::endl;
#endif

  bool isOdd = false;  // detId.chamber()%2;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b, B, L, t), isOdd));
  ME0Layer* layer = new ME0Layer(detId.layerId(), surf);
  return layer;
}

ME0EtaPartition* ME0GeometryBuilder::buildEtaPartition(DDFilteredView& fv, ME0DetId detId) const {
  LogTrace("ME0GeometryBuilder") << "buildEtaPartition " << fv.logicalPart().name().name() << " " << detId << std::endl;

  // EtaPartition specific parameter (nstrips and npads)
  DDValue numbOfStrips("nStrips");
  DDValue numbOfPads("nPads");
  std::vector<const DDsvalues_type*> specs(fv.specifics());
  double nStrips = 0., nPads = 0.;
  for (const auto& is : specs) {
    if (DDfetch(is, numbOfStrips))
      nStrips = numbOfStrips.doubles()[0];
    if (DDfetch(is, numbOfPads))
      nPads = numbOfPads.doubles()[0];
  }

  LogTrace("ME0GeometryBuilder") << ((nStrips == 0.) ? ("No nStrips found!!")
                                                     : ("Number of strips: " + std::to_string(nStrips)));
  LogTrace("ME0GeometryBuilder") << ((nPads == 0.) ? ("No nPads found!!")
                                                   : ("Number of pads: " + std::to_string(nPads)));

  // EtaPartition specific parameter (size)
  std::vector<double> dpar = fv.logicalPart().solid().parameters();
  double b = convertMmToCm(dpar[4]);  // half bottom edge
  double B = convertMmToCm(dpar[8]);  // half top edge
  double L = convertMmToCm(dpar[0]);  // half apothem
  double t = convertMmToCm(dpar[3]);  // half thickness

#ifdef EDM_ML_DEBUG
  LogTrace("ME0GeometryBuilder") << " name of logical part = " << fv.logicalPart().name().name() << std::endl;
  LogTrace("ME0GeometryBuilder") << " dpar is vector with size = " << dpar.size() << std::endl;
  for (unsigned int i = 0; i < dpar.size(); ++i) {
    LogTrace("ME0GeometryBuilder") << " dpar [" << i << "] = " << convertMmToCm(dpar[i]) << " cm " << std::endl;
  }
  LogTrace("ME0GeometryBuilder") << "size  b: " << b << "cm, B: " << B << "cm,  L: " << L << "cm, t: " << t << "cm "
                                 << std::endl;
#endif

  std::vector<float> pars;
  pars.emplace_back(b);
  pars.emplace_back(B);
  pars.emplace_back(L);
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);

  bool isOdd = false;  // detId.chamber()%2;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b, B, L, t), isOdd));
  std::string name = fv.logicalPart().name().name();
  ME0EtaPartitionSpecs* e_p_specs = new ME0EtaPartitionSpecs(GeomDetEnumerators::ME0, name, pars);

  ME0EtaPartition* etaPartition = new ME0EtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

ME0GeometryBuilder::ME0BoundPlane ME0GeometryBuilder::boundPlane(const DDFilteredView& fv,
                                                                 Bounds* bounds,
                                                                 bool isOddChamber) const {
  // extract the position
  const DDTranslation& trans(fv.translation());
  const Surface::PositionType posResult(
      float(convertMmToCm(trans.x())), float(convertMmToCm(trans.y())), float(convertMmToCm(trans.z())));

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
  Basic3DVector<float> newY(0., 0., 1.);
  Basic3DVector<float> newZ(0., 1., 0.);
  newY *= -1;

  rotResult.rotateAxes(newX, newY, newZ);

  return ME0BoundPlane(new BoundPlane(posResult, rotResult, bounds));
}

// dd4hep

ME0Geometry* ME0GeometryBuilder::buildGeometry(cms::DDFilteredView& fv, const MuonGeometryConstants& muonConstants) {
  ME0Geometry* geometry = new ME0Geometry();
  MuonGeometryNumbering mdddnum(muonConstants);
  ME0NumberingScheme me0Num(muonConstants);

  bool doChambers = fv.firstChild();
  //loop over chambers
  while (doChambers) {
    ME0DetId detId = ME0DetId(me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.history())));
    ME0DetId detIdCh = detId.chamberId();

    // build chamber
    ME0Chamber* me0Chamber = buildChamber(fv, detIdCh);
    geometry->add(me0Chamber);

    bool doLayers = fv.nextSibling();
    // loop over layers of the chamber
    while (doLayers) {
      ME0DetId detId = ME0DetId(me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.history())));
      ME0DetId detIdLa = detId.layerId();

      // build layer
      ME0Layer* me0Layer = buildLayer(fv, detIdLa);
      me0Chamber->add(me0Layer);
      geometry->add(me0Layer);

      fv.down();  // down to the first eta partion

      // build first eta partition
      ME0DetId detIdbis = ME0DetId(me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.history())));
      ME0EtaPartition* etaPart = buildEtaPartition(fv, detIdbis);
      me0Layer->add(etaPart);
      geometry->add(etaPart);

      bool doEtaParts = fv.sibling();
      // loop over the other eta partions

      while (doEtaParts) {
        ME0DetId detId = ME0DetId(me0Num.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.history())));
        // build other eta partitions
        ME0EtaPartition* etaPart = buildEtaPartition(fv, detId);
        me0Layer->add(etaPart);
        geometry->add(etaPart);

        doEtaParts = fv.sibling();
      }
      doLayers = fv.nextSibling();
    }
    fv.parent();
    doChambers = fv.firstChild();
  }

  return geometry;
}

ME0Chamber* ME0GeometryBuilder::buildChamber(cms::DDFilteredView& fv, ME0DetId detId) const {
  std::vector<double> dpar = fv.parameters();

  double L = dpar[3];
  double T = dpar[2];
  double b = dpar[0];
  double B = dpar[1];
  bool isOdd = false;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b, B, L, T), isOdd));
  ME0Chamber* chamber = new ME0Chamber(detId.chamberId(), surf);

  return chamber;
}

ME0Layer* ME0GeometryBuilder::buildLayer(cms::DDFilteredView& fv, ME0DetId detId) const {
  std::vector<double> dpar = fv.parameters();

  double L = dpar[3];
  double t = dpar[2];
  double b = dpar[0];
  double B = dpar[1];
  bool isOdd = false;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b, B, L, t), isOdd));
  ME0Layer* layer = new ME0Layer(detId.layerId(), surf);

  return layer;
}

ME0EtaPartition* ME0GeometryBuilder::buildEtaPartition(cms::DDFilteredView& fv, ME0DetId detId) const {
  //      auto nStrips = fv.get<double>("nStrips"); //it doesn't work
  //    auto nPads = fv.get<double>("nPads"); //it doesn't work

  auto nStrips = 384;  // from GEMSpecs
  auto nPads = 192;    // from GEMSpecs

  std::vector<double> dpar = fv.parameters();

  double b = dpar[0];
  double B = dpar[1];
  double L = dpar[3];
  double t = dpar[2];

  const std::vector<float> pars{float(dpar[0]), float(dpar[1]), float(dpar[3]), float(nStrips), float(nPads)};

  bool isOdd = false;
  ME0BoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(b, B, L, t), isOdd));

  std::string_view name = fv.name();

  ME0EtaPartitionSpecs* e_p_specs = new ME0EtaPartitionSpecs(GeomDetEnumerators::ME0, std::string(name), pars);

  ME0EtaPartition* etaPartition = new ME0EtaPartition(detId, surf, e_p_specs);

  return etaPartition;
}

ME0GeometryBuilder::ME0BoundPlane ME0GeometryBuilder::boundPlane(const cms::DDFilteredView& fv,
                                                                 Bounds* bounds,
                                                                 bool isOddChamber) const {
  // extract the position
  const Double_t* trans = fv.trans();
  Surface::PositionType posResult(trans[0], trans[1], trans[2]);

  // now the rotation
  DDRotationMatrix rotation;
  fv.rot(rotation);
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
  Basic3DVector<float> newY(0., 0., 1.);
  Basic3DVector<float> newZ(0., 1., 0.);
  newY *= -1;

  rotResult.rotateAxes(newX, newY, newZ);

  return ME0BoundPlane(new BoundPlane(posResult, rotResult, bounds));
}
