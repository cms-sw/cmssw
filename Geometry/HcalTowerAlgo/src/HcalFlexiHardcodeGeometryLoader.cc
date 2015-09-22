#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
//#define DebugLog
// ==============> Loader Itself <==========================

HcalFlexiHardcodeGeometryLoader::HcalFlexiHardcodeGeometryLoader(const edm::ParameterSet&) {

  MAX_HCAL_PHI = 72;
  DEGREE2RAD = M_PI / 180.;
}

CaloSubdetectorGeometry* HcalFlexiHardcodeGeometryLoader::load(const HcalTopology& fTopology, const HcalDDDRecConstants& hcons) {
  CaloSubdetectorGeometry* hcalGeometry = new HcalGeometry (fTopology);
  if( 0 == hcalGeometry->cornersMgr() ) hcalGeometry->allocateCorners ( fTopology.ncells() );
  if( 0 == hcalGeometry->parMgr() ) hcalGeometry->allocatePar (hcalGeometry->numberOfShapes(),
							       HcalGeometry::k_NumberOfParametersPerShape ) ;
  if (fTopology.mode() == HcalTopologyMode::H2) {  // TB geometry
    fillHBHO (hcalGeometry, makeHBCells(hcons), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHE (hcalGeometry, makeHECells_H2());
  } else { // regular geometry
    fillHBHO (hcalGeometry, makeHBCells(hcons), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHF (hcalGeometry, makeHFCells());
    fillHE (hcalGeometry, makeHECells(hcons));
  }
  return hcalGeometry;
}


// ----------> HB <-----------
std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters> HcalFlexiHardcodeGeometryLoader::makeHBCells (const HcalDDDRecConstants& hcons) {

  std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters> result;
  std::vector<std::pair<double,double> > gconsHB = hcons.getConstHBHE(0);
  std::vector<double> layerDepths;
  layerDepths.push_back(gconsHB[0].first-gconsHB[0].second);
  for (int i=0; i<17; ++i)
    layerDepths.push_back(gconsHB[i].first+gconsHB[i].second);
  std::vector<HcalDDDRecConstants::HcalEtaBin> etabins = hcons.getEtaBins(0);

#ifdef DebugLog
  std::cout << "FlexiGeometryLoader called for " << etabins.size() 
	    << " Eta Bins" << std::endl;
  for (unsigned int k=0; k<gconsHB.size(); ++k) {
    std::cout << "gconsHB[" << k << "] = " << gconsHB[k].first << ":"
	      << gconsHB[k].second << " LayerDepth[" << k << "] = "
	      << layerDepths[k] << std::endl;
  }
#endif
  for (unsigned int i=0; i<etabins.size(); ++i) {
    int iring = etabins[i].ieta;
    int nphi  = etabins[i].nPhi;
    int depth = etabins[i].depthStart;
    for (unsigned int k=0; k<etabins[i].layer.size(); ++k) {
      double rmin = layerDepths[etabins[i].layer[k].first-1];
      double rmax = layerDepths[etabins[i].layer[k].second];
#ifdef DebugLog
      std::cout << "HBRing " << iring << " eta " << etabins[i].etaMin << ":"
		<< etabins[i].etaMax << " depth " << depth << " R " << rmin
		<< ":" << rmax << " Phi 1:" << nphi << ":" << etabins[i].phi0
		<< ":" << etabins[i].dphi << " layer[" << k << "]: " 
		<< etabins[i].layer[k].first-1 << ":"
		<< etabins[i].layer[k].second << std::endl;
#endif
      result.push_back (HcalFlexiHardcodeGeometryLoader::HBHOCellParameters(iring, depth, 1, nphi, 1, etabins[i].phi0, etabins[i].dphi, rmin, rmax, etabins[i].etaMin, etabins[i].etaMax));
      depth++;
    }
  }
  return result;
}


// ----------> HO <-----------
std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters> HcalFlexiHardcodeGeometryLoader::makeHOCells () {
  const double HORMIN0 = 390.0;
  const double HORMIN1 = 412.6;
  const double HORMAX  = 413.6;
  const int    nCells  = 15;
  const double etamin[nCells] = {0.000,0.087,0.174, 0.261, 0.3395,0.435,0.522,
				 0.609,0.696,0.783, 0.873, 0.957, 1.044,1.131,
				 1.218};
  const double etamax[nCells] = {0.087,0.174,0.261, 0.3075,0.435, 0.522,0.609,
				 0.696,0.783,0.8494,0.957, 1.044, 1.131,1.218,
				 1.305};
  std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters> result;
  result.reserve (nCells);
  double dphi = 5*DEGREE2RAD;
  for (int i = 0; i < nCells; ++i) {
    double rmin = ((i < 4) ? HORMIN0 : HORMIN1);
    // eta, depth, firstPhi, stepPhi, deltaPhi, rMin, rMax, etaMin, etaMax
    result.push_back (HcalFlexiHardcodeGeometryLoader::HBHOCellParameters(i+1, 4, 1, 72, 1, 0, dphi, rmin, HORMAX, etamin[i], etamax[i]));
  }
  return result;
}


//
// Convert constants to appropriate cells
//
void HcalFlexiHardcodeGeometryLoader::fillHBHO (CaloSubdetectorGeometry* fGeometry, const std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters>& fCells, bool fHB) {

  for (size_t iCell = 0; iCell < fCells.size(); ++iCell) {
    const HcalFlexiHardcodeGeometryLoader::HBHOCellParameters& param = fCells[iCell];
    for (int iPhi = param.phiFirst; iPhi <= param.nPhi; iPhi += param.phiStep) {
      for (int iside = -1; iside <= 1; iside += 2) { // both detector sides are identical
	HcalDetId hid (fHB ? HcalBarrel : HcalOuter, param.eta*iside, iPhi, param.depth);
	float phiCenter = param.phiStart+(iPhi-0.5)*param.dphi; // middle of the cell
	float etaCenter = 0.5*(param.etaMin + param.etaMax);
	float x = param.rMin* cos (phiCenter);
	float y = param.rMin* sin (phiCenter);
	float z = iside * param.rMin * sinh(etaCenter);
	// make cell geometry
	GlobalPoint refPoint (x,y,z); // center of the cell's face
	std::vector<CCGFloat> cellParams;
	cellParams.reserve (5);
	cellParams.push_back (0.5 * (param.etaMax - param.etaMin)); // deta_half
	cellParams.push_back (0.5 * param.dphi);                    // dphi_half
	cellParams.push_back (0.5 * (param.rMax - param.rMin) * cosh (etaCenter)); // dr_half
	cellParams.push_back ( fabs( refPoint.eta() ) ) ;
	cellParams.push_back ( fabs( refPoint.z() ) ) ;
#ifdef DebugLog
	std::cout << "HcalFlexiHardcodeGeometryLoader::fillHBHO-> " << hid << " " << hid.rawId() << " " << std::hex << hid.rawId() << std::dec << " " << hid.ieta() << '/' << hid.iphi() << '/' << hid.depth() << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif
	fGeometry->newCell(refPoint,  refPoint,  refPoint, 
			   CaloCellGeometry::getParmPtr(cellParams, 
							fGeometry->parMgr(), 
							fGeometry->parVecVec()),
			   hid ) ;
      }
    }
  }
}


// ----------> HE <-----------
std::vector<HcalFlexiHardcodeGeometryLoader::HECellParameters> HcalFlexiHardcodeGeometryLoader::makeHECells (const HcalDDDRecConstants& hcons) {

  std::vector<HcalFlexiHardcodeGeometryLoader::HECellParameters> result;
  std::vector<std::pair<double,double> > gconsHE = hcons.getConstHBHE(1);
  std::vector<double> layerDepths;
#ifdef DebugLog
  std::cout << "HcalFlexiHardcodeGeometryLoader:HE with " << gconsHE.size() << " cells" << std::endl;
#endif
  if (gconsHE.size() > 0) {
    unsigned int istart = 1;
    layerDepths.push_back(gconsHE[istart].first-gconsHE[istart].second);
    for (unsigned int i=istart; i<gconsHE.size(); ++i)
      layerDepths.push_back(gconsHE[i].first+gconsHE[i].second);
    std::vector<HcalDDDRecConstants::HcalEtaBin> etabins = hcons.getEtaBins(1);

#ifdef DebugLog
    std::cout << "FlexiGeometryLoader called for HE with " << etabins.size() 
	      << " Eta Bins and " << layerDepths.size() << " depths" 
	      << std::endl;
    for (unsigned int i=0; i<layerDepths.size(); ++i)
      std::cout << " Depth[" << i << "] = " << layerDepths[i];
    std::cout << std::endl;
#endif
    for (unsigned int i=0; i<etabins.size(); ++i) {
      int    iring = etabins[i].ieta;
      int    nphi  = etabins[i].nPhi;
      int    depth = etabins[i].depthStart;
      double dphi  = etabins[i].dphi;
      int    units = int(((dphi*72)/(2*M_PI))+0.5);
      int    fioff = (units == 4) ? 3 : 1;
      nphi        *= units;
      for (unsigned int k=0; k<etabins[i].layer.size(); ++k) {
	int layf = etabins[i].layer[k].first-1;
	int layl = etabins[i].layer[k].second-1;
	double zmin = layerDepths[layf];
	double zmax = layerDepths[layl];
	if (zmin < 1.0) {
	  for (int k2=layf; k2<=layl; ++k2) {
	    if (layerDepths[k2] > 10) {
	      zmin = layerDepths[k2];
	      break;
	    }
	  }
	}
	if (zmin >= zmax) zmax = zmin+10.;
#ifdef DebugLog
	std::cout << "HERing " << iring << " eta " << etabins[i].etaMin << ":"
		  << etabins[i].etaMax << " depth " << depth << " Z " << zmin
		  << ":" << zmax << " Phi 1:" << nphi << ":" << etabins[i].phi0
		  << ":" << dphi << ":" << units << ":" << fioff  << " layer[" 
		  << k << "]: " << etabins[i].layer[k].first-1 << ":"
		  << etabins[i].layer[k].second-1 << std::endl;
#endif
	result.push_back(HcalFlexiHardcodeGeometryLoader::HECellParameters(iring, depth, fioff, nphi, units, etabins[i].phi0, dphi, zmin, zmax, etabins[i].etaMin, etabins[i].etaMax));
	depth++;
      }
    }
  }
  return result;
}


// ----------> HE @ H2 <-----------
std::vector <HcalFlexiHardcodeGeometryLoader::HECellParameters> HcalFlexiHardcodeGeometryLoader::makeHECells_H2 () {

  const double HEZMIN_H2 = 400.715;
  const double HEZMID_H2 = 436.285;
  const double HEZMAX_H2 = 541.885;
  const double dphi1 = 5*DEGREE2RAD;
  const double dphi2 = 2*dphi1;
    
  HcalFlexiHardcodeGeometryLoader::HECellParameters cells [] = {
    // eta, depth, firstPhi, nPhi, stepPhi, phiStart, deltaPhi, zMin, zMax, etaMin, etaMax
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 16, 3, 1, 8, 1, 0, dphi1, 409.885,   462.685,   1.305, 1.373),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 17, 1, 1, 8, 1, 0, dphi1, HEZMIN_H2, 427.485,   1.373, 1.444),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 17, 2, 1, 8, 1, 0, dphi1, 427.485,   506.685,   1.373, 1.444),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 18, 1, 1, 8, 1, 0, dphi1, HEZMIN_H2, HEZMID_H2, 1.444, 1.521),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 18, 2, 1, 8, 1, 0, dphi1, HEZMID_H2, 524.285,   1.444, 1.521),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 19, 1, 1, 8, 1, 0, dphi1, HEZMIN_H2, HEZMID_H2, 1.521, 1.603),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 19, 2, 1, 8, 1, 0, dphi1, HEZMID_H2, HEZMAX_H2, 1.521, 1.603),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 20, 1, 1, 8, 1, 0, dphi1, HEZMIN_H2, HEZMID_H2, 1.603, 1.693),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 20, 2, 1, 8, 1, 0, dphi1, HEZMID_H2, HEZMAX_H2, 1.603, 1.693),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 21, 1, 1, 8, 2, 0, dphi1, HEZMIN_H2, HEZMID_H2, 1.693, 1.79),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 21, 2, 1, 8, 2, 0, dphi1, HEZMID_H2, HEZMAX_H2, 1.693, 1.79),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 22, 1, 1, 8, 2, 0, dphi2, HEZMIN_H2, HEZMID_H2, 1.79, 1.88),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 22, 2, 1, 8, 2, 0, dphi2, HEZMID_H2, HEZMAX_H2, 1.79, 1.88),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 23, 1, 1, 8, 2, 0, dphi2, HEZMIN_H2, HEZMID_H2, 1.88, 1.98),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 23, 2, 1, 8, 2, 0, dphi2, HEZMID_H2, HEZMAX_H2, 1.88, 1.98),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 24, 1, 1, 8, 2, 0, dphi2, HEZMIN_H2, 418.685,   1.98, 2.09),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 24, 2, 1, 8, 2, 0, dphi2, 418.685,   HEZMID_H2, 1.98, 2.09),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 24, 3, 1, 8, 2, 0, dphi2, HEZMID_H2, HEZMAX_H2, 1.98, 2.09),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 25, 1, 1, 8, 2, 0, dphi2, HEZMIN_H2, 418.685,   2.09, 2.21),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 25, 2, 1, 8, 2, 0, dphi2, 418.685,   HEZMID_H2, 2.09, 2.21),
    HcalFlexiHardcodeGeometryLoader::HECellParameters ( 25, 3, 1, 8, 2, 0, dphi2, HEZMID_H2, HEZMAX_H2, 2.09, 2.21)
  };
  int nCells = sizeof(cells)/sizeof(HcalFlexiHardcodeGeometryLoader::HECellParameters);
  std::vector <HcalFlexiHardcodeGeometryLoader::HECellParameters> result;
  result.reserve (nCells);
  for (int i = 0; i < nCells; ++i) result.push_back (cells[i]);
  return result;
}

// ----------> HF <-----------
std::vector <HcalFlexiHardcodeGeometryLoader::HFCellParameters> HcalFlexiHardcodeGeometryLoader::makeHFCells () {

  const float HFZMIN1 = 1115.;
  const float HFZMIN2 = 1137.;
  const float HFZMAX = 1280.1;
    
  HcalFlexiHardcodeGeometryLoader::HFCellParameters cells [] = {
    // eta, depth, firstPhi, stepPhi, deltaPhi, zMin, zMax, rMin, rMax
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (29, 1, 1, 2, 10, HFZMIN1, HFZMAX,116.2,130.0),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (29, 2, 1, 2, 10, HFZMIN2, HFZMAX,116.2,130.0),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (30, 1, 1, 2, 10, HFZMIN1, HFZMAX, 97.5,116.2),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (30, 2, 1, 2, 10, HFZMIN2, HFZMAX, 97.5,116.2),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (31, 1, 1, 2, 10, HFZMIN1, HFZMAX, 81.8, 97.5),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (31, 2, 1, 2, 10, HFZMIN2, HFZMAX, 81.8, 97.5),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (32, 1, 1, 2, 10, HFZMIN1, HFZMAX, 68.6, 81.8),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (32, 2, 1, 2, 10, HFZMIN2, HFZMAX, 68.6, 81.8),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (33, 1, 1, 2, 10, HFZMIN1, HFZMAX, 57.6, 68.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (33, 2, 1, 2, 10, HFZMIN2, HFZMAX, 57.6, 68.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (34, 1, 1, 2, 10, HFZMIN1, HFZMAX, 48.3, 57.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (34, 2, 1, 2, 10, HFZMIN2, HFZMAX, 48.3, 57.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (35, 1, 1, 2, 10, HFZMIN1, HFZMAX, 40.6, 48.3),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (35, 2, 1, 2, 10, HFZMIN2, HFZMAX, 40.6, 48.3),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (36, 1, 1, 2, 10, HFZMIN1, HFZMAX, 34.0, 40.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (36, 2, 1, 2, 10, HFZMIN2, HFZMAX, 34.0, 40.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (37, 1, 1, 2, 10, HFZMIN1, HFZMAX, 28.6, 34.0),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (37, 2, 1, 2, 10, HFZMIN2, HFZMAX, 28.6, 34.0),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (38, 1, 1, 2, 10, HFZMIN1, HFZMAX, 24.0, 28.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (38, 2, 1, 2, 10, HFZMIN2, HFZMAX, 24.0, 28.6),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (39, 1, 1, 2, 10, HFZMIN1, HFZMAX, 20.1, 24.0),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (39, 2, 1, 2, 10, HFZMIN2, HFZMAX, 20.1, 24.0),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (40, 1, 3, 4, 20, HFZMIN1, HFZMAX, 16.9, 20.1),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (40, 2, 3, 4, 20, HFZMIN2, HFZMAX, 16.9, 20.1),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (41, 1, 3, 4, 20, HFZMIN1, HFZMAX, 12.5, 16.9),
    HcalFlexiHardcodeGeometryLoader::HFCellParameters (41, 2, 3, 4, 20, HFZMIN2, HFZMAX, 12.5, 16.9)
  };
  int nCells = sizeof(cells)/sizeof(HcalFlexiHardcodeGeometryLoader::HFCellParameters);
  std::vector <HcalFlexiHardcodeGeometryLoader::HFCellParameters> result;
  result.reserve (nCells);
  for (int i = 0; i < nCells; ++i) result.push_back (cells[i]);
  return result;
}
  
void HcalFlexiHardcodeGeometryLoader::fillHE (CaloSubdetectorGeometry* fGeometry, const std::vector <HcalFlexiHardcodeGeometryLoader::HECellParameters>& fCells) {

  for (size_t iCell = 0; iCell < fCells.size(); ++iCell) {
    const HcalFlexiHardcodeGeometryLoader::HECellParameters& param = fCells[iCell];
    int kPhi(param.phiFirst);
    for (int iPhi = param.phiFirst; iPhi <= param.nPhi; iPhi += param.phiStep, ++kPhi) {
      for (int iside = -1; iside <= 1; iside += 2) { // both detector sides are identical
	HcalDetId hid (HcalEndcap, param.eta*iside, iPhi, param.depth);
	float phiCenter = param.phiStart + (kPhi-0.5)*param.dphi; // middle of the cell
	float etaCenter = 0.5 * (param.etaMin + param.etaMax);

	float perp = param.zMin / sinh (etaCenter);
	float x = perp * cos (phiCenter);
	float y = perp * sin (phiCenter);
	float z = iside * param.zMin;
	// make cell geometry
	GlobalPoint refPoint (x,y,z); // center of the cell's face
	std::vector<CCGFloat> cellParams;
	cellParams.reserve (5);
	cellParams.push_back (0.5 * (param.etaMax - param.etaMin)); //deta_half
	cellParams.push_back (0.5 * param.dphi);  // dphi_half
	cellParams.push_back (-0.5 * (param.zMax - param.zMin) / tanh (etaCenter)); // dz_half, "-" means edges in Z
	cellParams.push_back ( fabs( refPoint.eta() ) ) ;
	cellParams.push_back ( fabs( refPoint.z() ) ) ;
#ifdef DebugLog
	std::cout << "HcalFlexiHardcodeGeometryLoader::fillHE-> " << hid << " " << hid.rawId() << " " << std::hex << hid.rawId() << std::dec << " " << hid.ieta() << '/' << hid.iphi() << '/' << hid.depth() << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif
	fGeometry->newCell(refPoint,  refPoint,  refPoint, 
			   CaloCellGeometry::getParmPtr(cellParams, 
							fGeometry->parMgr(), 
							fGeometry->parVecVec()),
			   hid ) ;
      }
    }
  }
}

void HcalFlexiHardcodeGeometryLoader::fillHF (CaloSubdetectorGeometry* fGeometry, const std::vector <HcalFlexiHardcodeGeometryLoader::HFCellParameters>& fCells) {

  for (size_t iCell = 0; iCell < fCells.size(); ++iCell) {
    const HcalFlexiHardcodeGeometryLoader::HFCellParameters& param = fCells[iCell];
//  int kPhi(0);
    for (int iPhi = param.phiFirst; iPhi <= MAX_HCAL_PHI; iPhi += param.phiStep) {
      for (int iside = -1; iside <= 1; iside += 2) { // both detector sides are identical
	HcalDetId hid (HcalForward, param.eta*iside, iPhi, param.depth);
	// middle of the cell
//      float phiCenter = ((kPhi + 0.5) * param.dphi) * DEGREE2RAD;
	float phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD;
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
	cellParams.push_back (0.5 * ( iEta - oEta )); // deta_half
	cellParams.push_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	cellParams.push_back (0.5 * (param.zMax - param.zMin)); // dz_half
	cellParams.push_back ( fabs( refPoint.eta()));
	cellParams.push_back ( fabs( refPoint.z() ) ) ;
#ifdef DebugLog
	std::cout << "HcalFlexiHardcodeGeometryLoader::fillHF-> " << hid << " " << hid.rawId() << " " << std::hex << hid.rawId() << std::dec << " " << hid.ieta() << '/' << hid.iphi() << '/' << hid.depth() << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif	
	fGeometry->newCell(refPoint,  refPoint,  refPoint, 
			   CaloCellGeometry::getParmPtr(cellParams, 
							fGeometry->parMgr(), 
							fGeometry->parVecVec()),
			   hid ) ;
      }
    }
  }
}
