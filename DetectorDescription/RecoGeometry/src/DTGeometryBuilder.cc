// -*- C++ -*-
//
// Package:    DetectorDescription/DTGeometryESProducer
// Class:      DTGeometryESProducer
// 
/**\class DTGeometryESProducer

 Description: DT Geometry builder from DD4hep

 Implementation:
     DT Geometry Builder iterates over a Detector Tree and
     retrvieves DT chambers, super layers, layers and wires.
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 16 Jan 2019 10:19:37 GMT
//
//
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"
#include "DetectorDescription/DDCMS/interface/MuonNumberingRcd.h"
#include "DetectorDescription/DDCMS/interface/MuonGeometryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDUnits.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DetectorDescription/RecoGeometry/interface/DTNumberingScheme.h"
#include "DetectorDescription/RecoGeometry/interface/DTGeometryBuilder.h"
#include "DD4hep/Detector.h"
#include "DD4hep/DetectorTools.h"
#include "DD4hep/VolumeProcessor.h"
#include "TGeoManager.h"
#include "TClass.h"

#include <memory>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <regex>

using namespace edm;
using namespace std;
using namespace cms;

namespace {
  
  string_view noNamespace(string_view input) {
    string_view v = input;
    auto first = v.find_first_of(":");
    v.remove_prefix(min(first+1, v.size()));
    return v;
  }
}

void
DTGeometryBuilder::buildGeometry(DDFilteredView& fview, dd4hep::Volume top,
				 const DDSpecPar& specpar,
				 DTGeometry& geom, const MuonNumbering& num) const {
  TGeoVolume *topVolume = top;
  TGeoIterator next(topVolume);
  next.SetType(0); // 0: DFS; 1: BFS
  TGeoNode *node;
  
  //find all Chamber nodes
  auto chamberSelection = specpar.vPathsTo(1);
  
  //find all Super Layer nodes
  auto superLayerSelection = specpar.tails(specpar.vPathsTo(2));
  
  //find all Layer nodes
  auto layerSelection = specpar.tails(specpar.vPathsTo(3));
  
  while((node = next())) {
    // If the node matches the Chamber selection
    if(fview.accepted(chamberSelection, noNamespace(node->GetVolume()->GetName()))) {
      // The matrix is absolute
      const TGeoMatrix *matrix = next.GetCurrentMatrix();
      const Double_t *rot = matrix->GetRotationMatrix();
      const Double_t *tr  = matrix->GetTranslation();
      
      // FIXME: Create an ExpanedNode here
      // int copyNo = node->GetNumber();
      DTChamber* chamber = buildChamber(fview,
					fview.extractParameters(node->GetVolume()),
					DDTranslation(tr[0], tr[1], tr[2]),
					DDRotationMatrix(rot[0], rot[1], rot[2],
							 rot[3], rot[4], rot[5],
							 rot[6], rot[7], rot[8]), num);
      // Loop on SLs
      TGeoNode *currentSLayer = nullptr;
      int nd = node->GetNdaughters();
      
      for(int i = 0; i < nd; ++i) {
	currentSLayer = (TGeoNode*)node->GetNodes()->At(i);
	
	if(fview.accepted(superLayerSelection, noNamespace(currentSLayer->GetVolume()->GetName()))) {
	  // FIXME: the matrix is relative
	  const TGeoMatrix *slMatrix = currentSLayer->GetMatrix();
	  const Double_t *slrot = slMatrix->GetRotationMatrix();
	  const Double_t *sltr  = slMatrix->GetTranslation();
	  
	  DTSuperLayer* sl = buildSuperLayer(chamber,
					     fview.extractParameters(currentSLayer->GetVolume()),
					     DDTranslation(sltr[0], sltr[1], sltr[2]),
					     DDRotationMatrix(slrot[0], slrot[1], slrot[2],
							      slrot[3], slrot[4], slrot[5],
							      slrot[6], slrot[7], slrot[8]), num);
	  // Loop on Layers
	  TGeoNode *currentLayer = nullptr;
	  int ndLs = currentSLayer->GetNdaughters();
	  for(int j = 0; j < ndLs; ++j) {
	    currentLayer = (TGeoNode*)currentSLayer->GetNodes()->At(j);
	    if(fview.accepted(layerSelection, noNamespace(currentLayer->GetVolume()->GetName()))) {
	      // FIXME: the matrix is relative
	      const TGeoMatrix *lMatrix = currentLayer->GetMatrix();
	      const Double_t *lrot = lMatrix->GetRotationMatrix();
	      const Double_t *ltr  = lMatrix->GetTranslation();
	      
	      DTLayer* l = buildLayer(sl,
				      fview.extractParameters(currentLayer->GetVolume()),
				      DDTranslation(ltr[0], ltr[1], ltr[2]),
				      DDRotationMatrix(lrot[0], lrot[1], lrot[2],
						       lrot[3], lrot[4], lrot[5],
						       lrot[6], lrot[7], lrot[8]), num, 1);
	      geom.add(l);
	    }
	  }
	  geom.add(sl);
	}
      }
      geom.add(chamber);
    }
  }
}

DTGeometryBuilder::RCPPlane
DTGeometryBuilder::plane(const DDTranslation& trans,
			 const DDRotationMatrix& rotation,
			 Bounds* bounds) const {

  const Surface::PositionType posResult(trans.x(),
					trans.y(),
					trans.z());
  DD3Vector x, y, z;
  rotation.GetComponents(x,y,z);
  
  Surface::RotationType rotResult(x.X(), x.Y(), x.Z(),
				  y.X(), y.Y(), y.Z(),
				  z.X(), z.Y(), z.Z()); 
  
  return RCPPlane(new Plane(posResult, rotResult, bounds));
}

DTChamber*
DTGeometryBuilder::buildChamber(const DDFilteredView& fview,
				const vector<double>& par,
				const DDTranslation& trans,
				const DDRotationMatrix& rotation,
				const MuonNumbering& muonConstants) const {
  MuonConstants cons = muonConstants.values;
  DTNumberingScheme dtnum(cons);
  int rawid = 574923776; //dtnum.getDetId(muonConstants.geoHistoryToBaseNumber(fview.geoHistory()));
  DTChamberId detId(rawid);
  
  float width = par[0];     // r-phi  dimension - different in different chambers
  float length = par[1];    // z      dimension - constant 125.55 cm
  float thickness = par[2]; // radial thickness - almost constant about 18 cm
  
  RCPPlane surf(plane(trans, rotation, new RectangularPlaneBounds(width, length, thickness)));
  
  DTChamber* chamber = new DTChamber(detId, surf);
  
  return chamber;
}

DTSuperLayer*
DTGeometryBuilder::buildSuperLayer(DTChamber* chamber,
				   const vector<double>& par,
				   const DDTranslation& trans,
				   const DDRotationMatrix& rotation,
				   const MuonNumbering&) const {
  // MuonDDDNumbering mdddnum(muonConstants);
  // DTNumberingScheme dtnum(muonConstants);
  // int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  int rawid = 574923776;
  DTSuperLayerId slId(rawid);
  
  float width = par[0];     // r-phi  dimension - changes in different chambers
  float length = par[1];    // z      dimension - constant 126.8 cm
  float thickness = par[2]; // radial thickness - almost constant about 20 cm
  
  // Ok this is the slayer position...
  RCPPlane surf(plane(trans, rotation, new RectangularPlaneBounds(width, length, thickness)));
  
  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);
  
  // add to the chamber
  chamber->add(slayer);
  
  return slayer;
}

DTLayer*
DTGeometryBuilder::buildLayer(DTSuperLayer* sl,
			      const vector<double>& par,
			      const DDTranslation& trans,
			      const DDRotationMatrix& rotation,
			      const MuonNumbering& num,
			      int copyno) const {
  // MuonDDDNumbering mdddnum(muonConstants);
  // DTNumberingScheme dtnum(muonConstants);
  // int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  int rawid = 574923776;
  DTLayerId layId(rawid);
  
  // Layer specific parameter (size)
  float width = par[0];     // r-phi  dimension - changes in different chambers
  float length = par[1];    // z      dimension - constant 126.8 cm
  float thickness = par[2]; // radial thickness - almost constant about 20 cm
  
  RCPPlane surf(plane(trans, rotation, new RectangularPlaneBounds(width, length, thickness)));
  
  // FIXME: // Loop on wires
  // bool doWire = fv.firstChild();
  int WCounter = 0;
  int firstWire = copyno;
  // par = extractParameters(fv);
  float wireLength = par[1];
  // while (doWire) {
  //   WCounter++;
  //   doWire = fv.nextSibling(); // next wire
  // }
  // //int lastWire=fv.copyno();
  DTTopology topology(firstWire, WCounter, wireLength);
  
  DTLayerType layerType;
  
  DTLayer* layer = new DTLayer(layId, surf, topology, layerType, sl);
  
  sl->add(layer);
  return layer;    
}

void
DTGeometryBuilder::build(DTGeometry& geom,
			 const DDDetector* det, 
			 const MuonNumbering& num,
			 const DDSpecParRefMap& refs) {
  
  dd4hep::DetElement world = det->description()->world();
  DDFilteredView fview(world.volume(), DDTranslation(), DDRotationMatrix());
  fview.mergedSpecifics(refs);
  fview.firstChild();
  // FIXME: Test for alternative iteration
  // buildGeometry(fview, world.volume(), *begin(refs)->second, geom, num);
  for(const auto& i: refs) {
    buildGeometry(fview, world.volume(), *i.second, geom, num);
  }
}
