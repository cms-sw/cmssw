#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <algorithm>

using CCGFloat = CaloCellGeometry::CCGFloat;
//#define DebugLog
// ==============> Loader Itself <==========================

HcalHardcodeGeometryLoader::HcalHardcodeGeometryLoader() {

  MAX_HCAL_PHI = 72;
  DEGREE2RAD = M_PI / 180.;
#ifdef DebugLog
  std::cout << "Instantiate HcalHardCodeGeometryLoader" << std::endl;
#endif
}

CaloSubdetectorGeometry* HcalHardcodeGeometryLoader::load(const HcalTopology& fTopology) {

  int maxEta = fTopology.lastHERing();
  m_segmentation.resize(maxEta);
  for (int i = 0; i < maxEta; i++) {
    fTopology.getDepthSegmentation(i+1,m_segmentation[i]);
#ifdef DebugLog
    std::cout << "Eta" << i+1;
    for (unsigned int k=0; k<m_segmentation[i].size(); ++k) {
      std::cout << " [" << k << "] " << m_segmentation[i][k];
    }
    std::cout << std::endl;
#endif
  }
  HcalGeometry* hcalGeometry = new HcalGeometry (fTopology);
  if( nullptr == hcalGeometry->cornersMgr() ) hcalGeometry->allocateCorners ( fTopology.ncells()+fTopology.getHFSize() );
  if( nullptr == hcalGeometry->parMgr() ) hcalGeometry->allocatePar (hcalGeometry->numberOfShapes(),
							       HcalGeometry::k_NumberOfParametersPerShape ) ;
  if (fTopology.mode() == HcalTopologyMode::H2) {  // TB geometry
    fillHBHO (hcalGeometry, makeHBCells(fTopology), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHE (hcalGeometry, makeHECells_H2());
 } else { // regular geometry
    fillHBHO (hcalGeometry, makeHBCells(fTopology), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHF (hcalGeometry, makeHFCells());
    fillHE (hcalGeometry, makeHECells(fTopology));
  }
  //fast insertion of valid ids requires sort at end
  hcalGeometry->sortValidIds();
  return hcalGeometry;
}


// ----------> HB <-----------
std::vector <HcalHardcodeGeometryLoader::HBHOCellParameters> HcalHardcodeGeometryLoader::makeHBCells (const HcalTopology & topology) {

  const float HBRMIN = 181.1;
  const float HBRMAX = 288.8;
    
  float normalDepths[2] = {HBRMIN, HBRMAX};
  float ring15Depths[3] = {HBRMIN, 258.4, HBRMAX};
  float ring16Depths[3] = {HBRMIN, 190.4, 232.6};
  float layerDepths[18] = {HBRMIN, 188.7, 194.7, 200.7, 206.7, 212.7, 218.7,
			   224.7, 230.7, 236.7, 242.7, 249.3, 255.9, 262.5,
			   269.1, 275.7, 282.3, HBRMAX};
  float slhcDepths[4]   = {HBRMIN, 214., 239., HBRMAX};
#ifdef DebugLog
  std::cout <<"FlexiGeometryLoader called for "<< topology.mode() << ":" << HcalTopologyMode::SLHC << std::endl;
#endif
  std::vector <HcalHardcodeGeometryLoader::HBHOCellParameters> result;
  for(int iring = 1; iring <= 16; ++iring) {
    std::vector<float> depths;
    if (topology.mode() != HcalTopologyMode::SLHC) {
      if (iring == 15) {
	for (float ring15Depth : ring15Depths) depths.emplace_back(ring15Depth);
      } else if (iring == 16) {
	for (float ring16Depth : ring16Depths) depths.emplace_back(ring16Depth);
      } else {
	for (float normalDepth : normalDepths) depths.emplace_back(normalDepth);
      }
    } else {
      if (m_segmentation.size() >= (unsigned int)(iring)) {
	int depth = m_segmentation[iring-1][0];
	depths.emplace_back(layerDepths[depth]);
	int layer = 1;
	for (unsigned int i=1; i<m_segmentation[iring-1].size(); ++i) {
	  if (depth != m_segmentation[iring-1][i]) {
	    depth = m_segmentation[iring-1][i];
	    layer = i;
	    if (iring != 16 || depth < 3)
	      depths.emplace_back(layerDepths[depth]);
	  }
	  if (i >= 17) break;
	}
	if (layer <= 17) depths.emplace_back(HBRMAX);
      } else {
	for (int i=0; i<4; ++i) {
	  if (iring != 16 || i < 3) {
	    depths.emplace_back(slhcDepths[i]);
	  }
	}
      }
    }
    unsigned int ndepth=depths.size()-1;
    unsigned int startingDepth=1;
    float etaMin=(iring-1)*0.087;
    float etaMax=iring*0.087;
    // topology.depthBinInformation(HcalBarrel, iring, ndepth, startingDepth);
#ifdef DebugLog
    std::cout << "HBRing " << iring << " eta " << etaMin << ":" << etaMax << " depths " << ndepth << ":" << startingDepth;
    for (unsigned int i=0; i<depths.size(); ++i) std::cout << ":" << depths[i];
    std::cout << "\n";
#endif
    for (unsigned int idepth = startingDepth; idepth <= ndepth; ++idepth) {
      float rmin = depths[idepth-1];
      float rmax = depths[idepth];
#ifdef DebugLog
      std::cout << "HB " << idepth << " R " << rmin << ":" << rmax << "\n";
#endif
      result.emplace_back(HcalHardcodeGeometryLoader::HBHOCellParameters(iring, (int)idepth, 1, 1, 5, rmin, rmax, etaMin, etaMax));
    }
  }
  return result;
}



// ----------> HO <-----------
std::vector <HcalHardcodeGeometryLoader::HBHOCellParameters> HcalHardcodeGeometryLoader::makeHOCells () {
  const float HORMIN0 = 390.0;
  const float HORMIN1 = 412.6;
  const float HORMAX = 413.6;
  
  HcalHardcodeGeometryLoader::HBHOCellParameters cells [] = {
    // eta, depth, firstPhi, stepPhi, deltaPhi, rMin, rMax, etaMin, etaMax
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 1, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*0, 0.087*1),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 2, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*1, 0.087*2),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 3, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*2, 0.087*3),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 4, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*3, 0.3075),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 5, 4, 1, 1, 5, HORMIN1, HORMAX, 0.3395,  0.087*5),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 6, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*5, 0.087*6),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 7, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*6, 0.087*7),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 8, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*7, 0.087*8),
    HcalHardcodeGeometryLoader::HBHOCellParameters ( 9, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*8, 0.087*9),
    HcalHardcodeGeometryLoader::HBHOCellParameters (10, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*9,  0.8494),
    HcalHardcodeGeometryLoader::HBHOCellParameters (11, 4, 1, 1, 5, HORMIN1, HORMAX, 0.873, 0.087*11),
    HcalHardcodeGeometryLoader::HBHOCellParameters (12, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*11, 0.087*12),
    HcalHardcodeGeometryLoader::HBHOCellParameters (13, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*12, 0.087*13),
    HcalHardcodeGeometryLoader::HBHOCellParameters (14, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*13, 0.087*14),
    HcalHardcodeGeometryLoader::HBHOCellParameters (15, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*14, 0.087*15)
  };
  int nCells = sizeof(cells)/sizeof(HcalHardcodeGeometryLoader::HBHOCellParameters);
  std::vector <HcalHardcodeGeometryLoader::HBHOCellParameters> result;
  result.reserve (nCells);
  for (int i = 0; i < nCells; ++i) result.emplace_back (cells[i]);
  return result;
}


//
// Convert constants to appropriate cells
//
void HcalHardcodeGeometryLoader::fillHBHO (HcalGeometry* fGeometry, const std::vector <HcalHardcodeGeometryLoader::HBHOCellParameters>& fCells, bool fHB) {

  fGeometry->increaseReserve(fCells.size());
  for (const auto & param : fCells) {
    for (int iPhi = param.phiFirst; iPhi <= MAX_HCAL_PHI; iPhi += param.phiStep) {
      for (int iside = -1; iside <= 1; iside += 2) { // both detector sides are identical
	HcalDetId hid (fHB ? HcalBarrel : HcalOuter, param.eta*iside, iPhi, param.depth);
	float phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD; // middle of the cell
	float etaCenter = 0.5*(param.etaMin + param.etaMax);
	float x = param.rMin* cos (phiCenter);
	float y = param.rMin* sin (phiCenter);
	float z = iside * param.rMin * sinh(etaCenter);
	// make cell geometry
	GlobalPoint refPoint (x,y,z); // center of the cell's face
	std::vector<CCGFloat> cellParams;
	cellParams.reserve (5);
	cellParams.emplace_back (0.5 * (param.etaMax - param.etaMin)); // deta_half
	cellParams.emplace_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	cellParams.emplace_back (0.5 * (param.rMax - param.rMin) * cosh (etaCenter)); // dr_half
	cellParams.emplace_back ( fabs( refPoint.eta() ) ) ;
	cellParams.emplace_back ( fabs( refPoint.z() ) ) ;
#ifdef DebugLog
	std::cout << "HcalHardcodeGeometryLoader::fillHBHO-> " << hid << hid.ieta() << '/' << hid.iphi() << '/' << hid.depth() << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif
	fGeometry->newCellFast(refPoint,  refPoint,  refPoint, 
			   CaloCellGeometry::getParmPtr(cellParams, 
							fGeometry->parMgr(), 
							fGeometry->parVecVec()),
			   hid ) ;
      }
    }
  }
}


// ----------> HE <-----------
std::vector<HcalHardcodeGeometryLoader::HECellParameters> HcalHardcodeGeometryLoader::makeHECells (const HcalTopology & topology) {

  std::vector <HcalHardcodeGeometryLoader::HECellParameters> result;
  const float HEZMIN = 400.458;
  const float HEZMID = 436.168;
  const float HEZMAX = 549.268;
  float normalDepths[3] = {HEZMIN, HEZMID, HEZMAX};
  float tripleDepths[4] = {HEZMIN, 418.768, HEZMID, HEZMAX};
  float slhcDepths[5]   = {HEZMIN, 418.768, HEZMID, 493., HEZMAX};
  float ring16Depths[2] = {418.768,470.968};
  float ring16slhcDepths[3] = {418.768, 450., 470.968};
  float ring17Depths[2] = {409.698,514.468};
  float ring17slhcDepths[5] = {409.698, 435., 460., 495., 514.468};
  float ring18Depths[3] = {391.883,427.468,540.568};
  float ring18slhcDepths[5] = {391.883, 439.,  467., 504. , 540.568};
  float etaBounds[] = {0.087*15, 0.087*16, 0.087*17, 0.087*18,  0.087*19,
		       1.74, 1.83,  1.93, 2.043, 2.172, 2.322, 2.500,
		       2.650, 2.868, 3.000};
  float layerDepths[19] = {HEZMIN, 408.718, 416.978, 425.248, 433.508, 441.768,
			   450.038,458.298, 466.558, 474.828, 483.088, 491.348,
			   499.618,507.878, 516.138, 524.398, 532.668, 540.928,
			   HEZMAX};

  // count by ring - 16
  for(int iringm16=0; iringm16 <= 13; ++iringm16) {
    int iring = iringm16 + 16;
    std::vector<float> depths;
    unsigned int startingDepth = 1;
    if (topology.mode() != HcalTopologyMode::SLHC) {
      if (iring == 16)     
	{for (float ring16Depth : ring16Depths) depths.emplace_back(ring16Depth); startingDepth = 3;}
      else if (iring == 17) 
	for (float ring17Depth : ring17Depths) depths.emplace_back(ring17Depth);
      else if (iring == 18) 
	for (float ring18Depth : ring18Depths) depths.emplace_back(ring18Depth);
      else if (iring == topology.lastHERing()) 
	for (int i=0; i<3; ++i) depths.emplace_back(tripleDepths[i]);
      else if (iring >= topology.firstHETripleDepthRing())
	for (float tripleDepth : tripleDepths) depths.emplace_back(tripleDepth);
      else
	for (float normalDepth : normalDepths) depths.emplace_back(normalDepth);
    } else {
      if (m_segmentation.size() >= (unsigned int)(iring)) {
	int depth = m_segmentation[iring-1][0];
	if (iring == 16)      depths.emplace_back(ring16Depths[0]);
	else if (iring == 17) depths.emplace_back(ring17Depths[0]);
	else if (iring == 18) depths.emplace_back(ring18Depths[0]);
	else                  depths.emplace_back(layerDepths[depth]);
	int layer = 1;
	float lastDepth = depths[0];
	for (unsigned int i=1; i<m_segmentation[iring-1].size(); ++i) {
	  if (depth != m_segmentation[iring-1][i]) {
	    depth = m_segmentation[iring-1][i];
	    layer = i;
	    if (layerDepths[depth] > lastDepth && (iring != 16 || depth > 3)) {
	      depths.emplace_back(layerDepths[depth]);
	      lastDepth = layerDepths[depth];
	    }
	  }
	}
	if (layer <= 17) depths.emplace_back(HEZMAX);
	if (iring == 16) startingDepth = 3;
      } else {
	if (iring == 16)     {for (float ring16slhcDepth : ring16slhcDepths) depths.emplace_back(ring16slhcDepth); startingDepth = 3;}
	else if (iring == 17) for (float ring17slhcDepth : ring17slhcDepths) depths.emplace_back(ring17slhcDepth);
	else if (iring == 18) for (float ring18slhcDepth : ring18slhcDepths) depths.emplace_back(ring18slhcDepth);
	else                  for (float slhcDepth : slhcDepths) depths.emplace_back(slhcDepth);
      }
    }
    float etamin = etaBounds[iringm16];
    float etamax = etaBounds[iringm16+1];
    unsigned int ndepth = depths.size()-1;
    //    topology.depthBinInformation(HcalEndcap, iring, ndepth, startingDepth);
#ifdef DebugLog
    std::cout << "HERing " << iring << " eta " << etamin << ":" << etamax << " depths " << ndepth << ":" << startingDepth;
    for (unsigned int i=0; i<depths.size(); ++i) std::cout << ":" << depths[i];
    std::cout << "\n";
#endif
    for (unsigned int idepth = 0; idepth < ndepth; ++idepth) {
      int depthIndex = (int)(idepth + startingDepth);
      float zmin = depths[idepth];
      float zmax = depths[idepth+1];
      if (depthIndex <= 7) {
#ifdef DebugLog
	std::cout << "HE Depth " << idepth << ":" << depthIndex << " Z " << zmin << ":" << zmax << "\n";
#endif
	int stepPhi = (iring >= topology.firstHEDoublePhiRing() ? 2 : 1);
	int deltaPhi =  (iring >= topology.firstHEDoublePhiRing() ? 10 : 5);
	if (topology.mode() != HcalTopologyMode::SLHC &&
	    iring == topology.lastHERing()-1 && idepth == ndepth-1) {
#ifdef DebugLog
	  std::cout << "HE iEta " << iring << " Depth " << depthIndex << " Eta " << etamin << ":" << etaBounds[iringm16+2] << std::endl;
#endif
	  result.emplace_back(HcalHardcodeGeometryLoader::HECellParameters(iring, depthIndex, 1, stepPhi, deltaPhi, zmin, zmax, etamin, etaBounds[iringm16+2]));
	} else {
#ifdef DebugLog
	  std::cout << "HE iEta " << iring << " Depth " << depthIndex << " Eta " << etamin << ":" << etamax << std::endl;
#endif
	  result.emplace_back(HcalHardcodeGeometryLoader::HECellParameters(iring, depthIndex, 1, stepPhi, deltaPhi, zmin, zmax, etamin, etamax));
	}
      }
    }
  }

  return result;
}


// ----------> HE @ H2 <-----------
std::vector <HcalHardcodeGeometryLoader::HECellParameters> HcalHardcodeGeometryLoader::makeHECells_H2 () {

  const float HEZMIN_H2 = 400.715;
  const float HEZMID_H2 = 436.285;
  const float HEZMAX_H2 = 541.885;
    
  HcalHardcodeGeometryLoader::HECellParameters cells [] = {
    // eta, depth, firstPhi, stepPhi, deltaPhi, zMin, zMax, etaMin, etaMax
    HcalHardcodeGeometryLoader::HECellParameters ( 16, 3, 1, 1, 5, 409.885,   462.685,   1.305, 1.373),
    HcalHardcodeGeometryLoader::HECellParameters ( 17, 1, 1, 1, 5, HEZMIN_H2, 427.485,   1.373, 1.444),
    HcalHardcodeGeometryLoader::HECellParameters ( 17, 2, 1, 1, 5, 427.485,   506.685,   1.373, 1.444),
    HcalHardcodeGeometryLoader::HECellParameters ( 18, 1, 1, 1, 5, HEZMIN_H2, HEZMID_H2, 1.444, 1.521),
    HcalHardcodeGeometryLoader::HECellParameters ( 18, 2, 1, 1, 5, HEZMID_H2, 524.285,   1.444, 1.521),
    HcalHardcodeGeometryLoader::HECellParameters ( 19, 1, 1, 1, 5, HEZMIN_H2, HEZMID_H2, 1.521, 1.603),
    HcalHardcodeGeometryLoader::HECellParameters ( 19, 2, 1, 1, 5, HEZMID_H2, HEZMAX_H2, 1.521, 1.603),
    HcalHardcodeGeometryLoader::HECellParameters ( 20, 1, 1, 1, 5, HEZMIN_H2, HEZMID_H2, 1.603, 1.693),
    HcalHardcodeGeometryLoader::HECellParameters ( 20, 2, 1, 1, 5, HEZMID_H2, HEZMAX_H2, 1.603, 1.693),
    HcalHardcodeGeometryLoader::HECellParameters ( 21, 1, 1, 2, 5, HEZMIN_H2, HEZMID_H2, 1.693, 1.79),
    HcalHardcodeGeometryLoader::HECellParameters ( 21, 2, 1, 2, 5, HEZMID_H2, HEZMAX_H2, 1.693, 1.79),
    HcalHardcodeGeometryLoader::HECellParameters ( 22, 1, 1, 2,10, HEZMIN_H2, HEZMID_H2, 1.79, 1.88),
    HcalHardcodeGeometryLoader::HECellParameters ( 22, 2, 1, 2,10, HEZMID_H2, HEZMAX_H2, 1.79, 1.88),
    HcalHardcodeGeometryLoader::HECellParameters ( 23, 1, 1, 2,10, HEZMIN_H2, HEZMID_H2, 1.88, 1.98),
    HcalHardcodeGeometryLoader::HECellParameters ( 23, 2, 1, 2,10, HEZMID_H2, HEZMAX_H2, 1.88, 1.98),
    HcalHardcodeGeometryLoader::HECellParameters ( 24, 1, 1, 2,10, HEZMIN_H2, 418.685,   1.98, 2.09),
    HcalHardcodeGeometryLoader::HECellParameters ( 24, 2, 1, 2,10, 418.685,   HEZMID_H2, 1.98, 2.09),
    HcalHardcodeGeometryLoader::HECellParameters ( 24, 3, 1, 2,10, HEZMID_H2, HEZMAX_H2, 1.98, 2.09),
    HcalHardcodeGeometryLoader::HECellParameters ( 25, 1, 1, 2,10, HEZMIN_H2, 418.685,   2.09, 2.21),
    HcalHardcodeGeometryLoader::HECellParameters ( 25, 2, 1, 2,10, 418.685,   HEZMID_H2, 2.09, 2.21),
    HcalHardcodeGeometryLoader::HECellParameters ( 25, 3, 1, 2,10, HEZMID_H2, HEZMAX_H2, 2.09, 2.21)
  };
  int nCells = sizeof(cells)/sizeof(HcalHardcodeGeometryLoader::HECellParameters);
  std::vector <HcalHardcodeGeometryLoader::HECellParameters> result;
  result.reserve (nCells);
  for (int i = 0; i < nCells; ++i) result.emplace_back (cells[i]);
  return result;
}

// ----------> HF <-----------
std::vector <HcalHardcodeGeometryLoader::HFCellParameters> HcalHardcodeGeometryLoader::makeHFCells () {

  const float HFZMIN1 = 1115.;
  const float HFZMIN2 = 1137.;
  const float HFZMAX = 1280.1;
    
  HcalHardcodeGeometryLoader::HFCellParameters cells [] = {
    // eta, depth, firstPhi, stepPhi, deltaPhi, zMin, zMax, rMin, rMax
    HcalHardcodeGeometryLoader::HFCellParameters (29, 1, 1, 2, 10, HFZMIN1, HFZMAX,116.2,130.0),
    HcalHardcodeGeometryLoader::HFCellParameters (29, 2, 1, 2, 10, HFZMIN2, HFZMAX,116.2,130.0),
    HcalHardcodeGeometryLoader::HFCellParameters (30, 1, 1, 2, 10, HFZMIN1, HFZMAX, 97.5,116.2),
    HcalHardcodeGeometryLoader::HFCellParameters (30, 2, 1, 2, 10, HFZMIN2, HFZMAX, 97.5,116.2),
    HcalHardcodeGeometryLoader::HFCellParameters (31, 1, 1, 2, 10, HFZMIN1, HFZMAX, 81.8, 97.5),
    HcalHardcodeGeometryLoader::HFCellParameters (31, 2, 1, 2, 10, HFZMIN2, HFZMAX, 81.8, 97.5),
    HcalHardcodeGeometryLoader::HFCellParameters (32, 1, 1, 2, 10, HFZMIN1, HFZMAX, 68.6, 81.8),
    HcalHardcodeGeometryLoader::HFCellParameters (32, 2, 1, 2, 10, HFZMIN2, HFZMAX, 68.6, 81.8),
    HcalHardcodeGeometryLoader::HFCellParameters (33, 1, 1, 2, 10, HFZMIN1, HFZMAX, 57.6, 68.6),
    HcalHardcodeGeometryLoader::HFCellParameters (33, 2, 1, 2, 10, HFZMIN2, HFZMAX, 57.6, 68.6),
    HcalHardcodeGeometryLoader::HFCellParameters (34, 1, 1, 2, 10, HFZMIN1, HFZMAX, 48.3, 57.6),
    HcalHardcodeGeometryLoader::HFCellParameters (34, 2, 1, 2, 10, HFZMIN2, HFZMAX, 48.3, 57.6),
    HcalHardcodeGeometryLoader::HFCellParameters (35, 1, 1, 2, 10, HFZMIN1, HFZMAX, 40.6, 48.3),
    HcalHardcodeGeometryLoader::HFCellParameters (35, 2, 1, 2, 10, HFZMIN2, HFZMAX, 40.6, 48.3),
    HcalHardcodeGeometryLoader::HFCellParameters (36, 1, 1, 2, 10, HFZMIN1, HFZMAX, 34.0, 40.6),
    HcalHardcodeGeometryLoader::HFCellParameters (36, 2, 1, 2, 10, HFZMIN2, HFZMAX, 34.0, 40.6),
    HcalHardcodeGeometryLoader::HFCellParameters (37, 1, 1, 2, 10, HFZMIN1, HFZMAX, 28.6, 34.0),
    HcalHardcodeGeometryLoader::HFCellParameters (37, 2, 1, 2, 10, HFZMIN2, HFZMAX, 28.6, 34.0),
    HcalHardcodeGeometryLoader::HFCellParameters (38, 1, 1, 2, 10, HFZMIN1, HFZMAX, 24.0, 28.6),
    HcalHardcodeGeometryLoader::HFCellParameters (38, 2, 1, 2, 10, HFZMIN2, HFZMAX, 24.0, 28.6),
    HcalHardcodeGeometryLoader::HFCellParameters (39, 1, 1, 2, 10, HFZMIN1, HFZMAX, 20.1, 24.0),
    HcalHardcodeGeometryLoader::HFCellParameters (39, 2, 1, 2, 10, HFZMIN2, HFZMAX, 20.1, 24.0),
    HcalHardcodeGeometryLoader::HFCellParameters (40, 1, 3, 4, 20, HFZMIN1, HFZMAX, 16.9, 20.1),
    HcalHardcodeGeometryLoader::HFCellParameters (40, 2, 3, 4, 20, HFZMIN2, HFZMAX, 16.9, 20.1),
    HcalHardcodeGeometryLoader::HFCellParameters (41, 1, 3, 4, 20, HFZMIN1, HFZMAX, 12.5, 16.9),
    HcalHardcodeGeometryLoader::HFCellParameters (41, 2, 3, 4, 20, HFZMIN2, HFZMAX, 12.5, 16.9)
  };
  int nCells = sizeof(cells)/sizeof(HcalHardcodeGeometryLoader::HFCellParameters);
  std::vector <HcalHardcodeGeometryLoader::HFCellParameters> result;
  result.reserve (nCells);
  for (int i = 0; i < nCells; ++i) result.emplace_back (cells[i]);
  return result;
}
  
void HcalHardcodeGeometryLoader::fillHE (HcalGeometry* fGeometry, const std::vector <HcalHardcodeGeometryLoader::HECellParameters>& fCells) {

  fGeometry->increaseReserve(fCells.size());
  for (const auto & param : fCells) {
    for (int iPhi = param.phiFirst; iPhi <= MAX_HCAL_PHI; iPhi += param.phiStep) {
      for (int iside = -1; iside <= 1; iside += 2) { // both detector sides are identical
	HcalDetId hid (HcalEndcap, param.eta*iside, iPhi, param.depth);
	float phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD; // middle of the cell
	float etaCenter = 0.5 * (param.etaMin + param.etaMax);

	float perp = param.zMin / sinh (etaCenter);
	float x = perp * cos (phiCenter);
	float y = perp * sin (phiCenter);
	float z = iside * param.zMin;
	// make cell geometry
	GlobalPoint refPoint (x,y,z); // center of the cell's face
	std::vector<CCGFloat> cellParams;
	cellParams.reserve (5);
	cellParams.emplace_back (0.5 * (param.etaMax - param.etaMin)); //deta_half
	cellParams.emplace_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	cellParams.emplace_back (-0.5 * (param.zMax - param.zMin) / tanh (etaCenter)); // dz_half, "-" means edges in Z
	cellParams.emplace_back ( fabs( refPoint.eta() ) ) ;
	cellParams.emplace_back ( fabs( refPoint.z() ) ) ;
#ifdef DebugLog
	std::cout << "HcalHardcodeGeometryLoader::fillHE-> " << hid << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif
	fGeometry->newCellFast(refPoint,  refPoint,  refPoint, 
			   CaloCellGeometry::getParmPtr(cellParams, 
							fGeometry->parMgr(), 
							fGeometry->parVecVec()),
			   hid ) ;
      }
    }
  }
}

void HcalHardcodeGeometryLoader::fillHF (HcalGeometry* fGeometry, const std::vector <HcalHardcodeGeometryLoader::HFCellParameters>& fCells) {

  fGeometry->increaseReserve(fCells.size());
  for (const auto & param : fCells) {
    for (int iPhi = param.phiFirst; iPhi <= MAX_HCAL_PHI; iPhi += param.phiStep) {
      for (int iside = -1; iside <= 1; iside += 2) { // both detector sides are identical
	HcalDetId hid (HcalForward, param.eta*iside, iPhi, param.depth);
	float phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD; // middle of the cell
	GlobalPoint inner (param.rMin, 0, param.zMin);
	GlobalPoint outer (param.rMax, 0, param.zMin);
	float iEta = inner.eta();
	float oEta = outer.eta();
	float etaCenter = 0.5 * ( iEta + oEta );
	
	float perp = param.zMin / sinh (etaCenter);
	float x = perp * cos (phiCenter);
	float y = perp * sin (phiCenter);
	float z = iside * param.zMin;
	// make cell geometry
	GlobalPoint refPoint (x,y,z); // center of the cell's face
	std::vector<CCGFloat> cellParams;
	cellParams.reserve (5);
	cellParams.emplace_back (0.5 * ( iEta - oEta )); // deta_half
	cellParams.emplace_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	cellParams.emplace_back (0.5 * (param.zMax - param.zMin)); // dz_half
	cellParams.emplace_back ( fabs( refPoint.eta()));
	cellParams.emplace_back ( fabs( refPoint.z() ) ) ;
#ifdef DebugLog
	std::cout << "HcalHardcodeGeometryLoader::fillHF-> " << hid << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif	
	fGeometry->newCellFast(refPoint,  refPoint,  refPoint, 
			   CaloCellGeometry::getParmPtr(cellParams, 
							fGeometry->parMgr(), 
							fGeometry->parVecVec()),
			   hid ) ;
      }
    }
  }
}
