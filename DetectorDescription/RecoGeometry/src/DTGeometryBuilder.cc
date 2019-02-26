// -*- C++ -*-
//
// Package:    DetectorDescription/DTGeometryBuilder
// Class:      DTGeometryBuilder
// 
/**\class DTGeometryBuilder

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

#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"
#include "DetectorDescription/DDCMS/interface/MuonNumberingRcd.h"
#include "DetectorDescription/DDCMS/interface/MuonGeometryRcd.h"
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
DTGeometryBuilder::buildGeometry(DDFilteredView& fview,
				 const DDSpecPar& specpar,
				 DTGeometry& geom, const MuonNumbering& num) const {
  TGeoVolume *topVolume = fview.geoHistory().back().volume;
  TGeoIterator next(topVolume);
  next.SetType(0); // 0: DFS; 1: BFS
  TGeoNode *node;
  
  //find all Chamber nodes
  auto const& chamberSelection = fview.vPathsTo(specpar, 1);
  
  //find all Super Layer nodes
  auto const& superLayerSelection = fview.tails(fview.vPathsTo(specpar, 2));
  
  //find all Layer nodes
  auto const& layerSelection = fview.tails(fview.vPathsTo(specpar, 3));
  
  // Not used:
  // auto const& type = specpar.strValue("Type");
  // auto const& FEPos = specpar.strValue("FEPos");
  
  while((node = next())) {
    // If the node matches the Chamber selection
    if(fview.accepted(chamberSelection, noNamespace(node->GetVolume()->GetName()))) {
      // The matrix is absolute
      const TGeoMatrix *matrix = next.GetCurrentMatrix();
      const Double_t *rot = matrix->GetRotationMatrix();
      const Double_t *tr  = matrix->GetTranslation();
      
      // FIXME: Create an ExpanedNode here
      int nd = node->GetNdaughters();
      TString path;
      next.GetPath(path);
      fview.checkPath(path, node);

      DDTranslation chTr = DDTranslation(tr[0], tr[1], tr[2]);
      DDRotationMatrix chRot = DDRotationMatrix(rot[0], rot[1], rot[2],
						rot[3], rot[4], rot[5],
						rot[6], rot[7], rot[8]);
      DTChamber* chamber = buildChamber(fview,
					chTr, chRot,
					num);
      // Loop on SLs
      TGeoNode *currentSLayer = nullptr;
      
      for(int i = 0; i < nd; ++i) {
	currentSLayer = static_cast<TGeoNode*>(node->GetNodes()->At(i));
	
	if(fview.accepted(superLayerSelection, noNamespace(currentSLayer->GetVolume()->GetName()))) {	  
	  // The matrix is relative
	  const TGeoMatrix *slMatrix = currentSLayer->GetMatrix();
	  const Double_t *slrot = slMatrix->GetRotationMatrix();
	  const Double_t *sltr  = slMatrix->GetTranslation();
	  // This matrix is absolute
	  DDTranslation slTr = chTr + (chRot * DDTranslation(sltr[0], sltr[1], sltr[2]));
	  DDRotationMatrix slRot = chRot * DDRotationMatrix(slrot[0], slrot[1], slrot[2],
							    slrot[3], slrot[4], slrot[5],
							    slrot[6], slrot[7], slrot[8]);
	    
	  fview.checkNode(currentSLayer);
	  
	  DTSuperLayer* sl = buildSuperLayer(fview, chamber,
					     slTr, slRot,
					     num);
	  // Loop on Layers
	  TGeoNode *currentLayer = nullptr;
	  int ndLs = currentSLayer->GetNdaughters();
	  for(int j = 0; j < ndLs; ++j) {
	    currentLayer = static_cast<TGeoNode*>(currentSLayer->GetNodes()->At(j));
	    if(fview.accepted(layerSelection, noNamespace(currentLayer->GetVolume()->GetName()))) {
	      // The matrix is relative
	      const TGeoMatrix *lMatrix = currentLayer->GetMatrix();
	      const Double_t *lrot = lMatrix->GetRotationMatrix();
	      const Double_t *ltr  = lMatrix->GetTranslation();

	      // This matrix is absolute
	      DDTranslation lTr = slTr + (slRot * DDTranslation(ltr[0], ltr[1], ltr[2]));
	      DDRotationMatrix lRot = slRot * DDRotationMatrix(lrot[0], lrot[1], lrot[2],
							       lrot[3], lrot[4], lrot[5],
							       lrot[6], lrot[7], lrot[8]);
	    
	      fview.checkNode(currentLayer);
   
	      DTLayer* l = buildLayer(fview, sl,
				      lTr, lRot,
				      num);
	      fview.unCheckNode();
	      geom.add(l);
	    }
	  }
	  fview.unCheckNode();
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
				const DDTranslation& trans,
				const DDRotationMatrix& rotation,
				const MuonNumbering& muonConstants) const {
  MuonConstants cons = muonConstants.values;
  DTNumberingScheme dtnum(cons);
  
  int rawid = dtnum.getDetId(muonConstants.geoHistoryToBaseNumber(fview.nodes));
  DTChamberId detId(rawid);
  auto const& par = fview.extractParameters();
  // par[0] r-phi  dimension - different in different chambers
  // par[1] z      dimension - constant 125.55 cm
  // par[2] radial thickness - almost constant about 18 cm  
  RCPPlane surf(plane(trans, rotation, new RectangularPlaneBounds(par[0], par[1], par[2])));
  
  DTChamber* chamber = new DTChamber(detId, surf);
  
  return chamber;
}

DTSuperLayer*
DTGeometryBuilder::buildSuperLayer(const DDFilteredView& fview,
				   DTChamber* chamber,
				   const DDTranslation& trans,
				   const DDRotationMatrix& rotation,
				   const MuonNumbering& muonConstants) const {
  MuonConstants cons = muonConstants.values;
  DTNumberingScheme dtnum(cons);

  int rawid = dtnum.getDetId(muonConstants.geoHistoryToBaseNumber(fview.nodes));
  DTSuperLayerId slId(rawid);
  
  auto const& par = fview.extractParameters();
  // par[0] r-phi  dimension - changes in different chambers
  // par[1] z      dimension - constant 126.8 cm
  // par[2] radial thickness - almost constant about 20 cm
  
  // Ok this is the slayer position...
  RCPPlane surf(plane(trans, rotation, new RectangularPlaneBounds(par[0], par[1], par[2])));
  
  DTSuperLayer* slayer = new DTSuperLayer(slId, surf, chamber);
  
  // add to the chamber
  chamber->add(slayer);
  
  return slayer;
}

DTLayer*
DTGeometryBuilder::buildLayer(const DDFilteredView& fview,
			      DTSuperLayer* sl,
			      const DDTranslation& trans,
			      const DDRotationMatrix& rotation,
			      const MuonNumbering& muonConstants) const {
  MuonConstants cons = muonConstants.values;
  DTNumberingScheme dtnum(cons);

  int rawid = dtnum.getDetId(muonConstants.geoHistoryToBaseNumber(fview.nodes));
  DTLayerId layId(rawid);
  
  auto const& par = fview.extractParameters();
  // Layer specific parameter (size)
  // par[0] r-phi  dimension - changes in different chambers
  // par[1] z      dimension - constant 126.8 cm
  // par[2] radial thickness - almost constant about 20 cm
  RCPPlane surf(plane(trans, rotation, new RectangularPlaneBounds(par[0], par[1], par[2])));
  
  // FIXME: // Loop on wires
  // bool doWire = fv.firstChild();
  int WCounter = 0;
  int firstWire = 1; //FIXME: copyno;
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
			 const cms::DDDetector* det,
			 const MuonNumbering& num,
			 const DDSpecParRefs& refs) {
  DDFilteredView fview(det);

  for(const auto& i: refs) {
    buildGeometry(fview, *i, geom, num);
  }
}
