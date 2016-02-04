#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

typedef CaloCellGeometry::CCGFloat CCGFloat ;

namespace {
  const int MAX_HCAL_PHI = 72;
  const float DEGREE2RAD = M_PI / 180.;

  // Parameter objects

  struct HBHOCellParameters {
    HBHOCellParameters (int f_eta, int f_depth, int f_phiFirst, int f_phiStep, int f_dPhi, float f_rMin, float f_rMax, float f_etaMin, float f_etaMax)
      : eta(f_eta), depth(f_depth), phiFirst(f_phiFirst), phiStep(f_phiStep), dphi(f_dPhi), rMin(f_rMin), rMax(f_rMax), etaMin(f_etaMin), etaMax(f_etaMax)
    {}
 
    int eta;
    int depth;
    int phiFirst;
    int phiStep;
    int dphi;
    float rMin;
    float rMax;
    float etaMin;
    float etaMax;
  };

  struct HECellParameters {
    HECellParameters (int f_eta, int f_depth, int f_phiFirst, int f_phiStep, int f_dPhi, float f_zMin, float f_zMax, float f_etaMin, float f_etaMax)
      : eta(f_eta), depth(f_depth), phiFirst(f_phiFirst), phiStep(f_phiStep), dphi(f_dPhi), zMin(f_zMin), zMax(f_zMax), etaMin(f_etaMin), etaMax(f_etaMax)
    {}
 
    int eta;
    int depth;
    int phiFirst;
    int phiStep;
    int dphi;
    float zMin;
    float zMax;
    float etaMin;
    float etaMax;
  };

  struct HFCellParameters {
    HFCellParameters (int f_eta, int f_depth, int f_phiFirst, int f_phiStep, int f_dPhi, float f_zMin, float f_zMax, float f_rMin, float f_rMax)
      : eta(f_eta), depth(f_depth), phiFirst(f_phiFirst), phiStep(f_phiStep), dphi(f_dPhi), zMin(f_zMin), zMax(f_zMax), rMin(f_rMin), rMax(f_rMax)
    {}
 
    int eta;
    int depth;
    int phiFirst;
    int phiStep;
    int dphi;
    float zMin;
    float zMax;
    float rMin;
    float rMax;
  };



  // ----------> HB <-----------
  std::vector <HBHOCellParameters> makeHBCells () {
    const float HBRMIN = 181.1;
    const float HBRMAX = 288.8;
    
    HBHOCellParameters cells [] = {
      // eta, depth, firstPhi, stepPhi, deltaPhi, rMin, rMax, etaMin, etaMax
      HBHOCellParameters ( 1, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*0, 0.087*1),
      HBHOCellParameters ( 2, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*1, 0.087*2),
      HBHOCellParameters ( 3, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*2, 0.087*3),
      HBHOCellParameters ( 4, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*3, 0.087*4),
      HBHOCellParameters ( 5, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*4, 0.087*5),
      HBHOCellParameters ( 6, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*5, 0.087*6),
      HBHOCellParameters ( 7, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*6, 0.087*7),
      HBHOCellParameters ( 8, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*7, 0.087*8),
      HBHOCellParameters ( 9, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*8, 0.087*9),
      HBHOCellParameters (10, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*9, 0.087*10),
      HBHOCellParameters (11, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*10, 0.087*11),
      HBHOCellParameters (12, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*11, 0.087*12),
      HBHOCellParameters (13, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*12, 0.087*13),
      HBHOCellParameters (14, 1, 1, 1, 5, HBRMIN, HBRMAX, 0.087*13, 0.087*14),
      HBHOCellParameters (15, 1, 1, 1, 5, HBRMIN, 258.4,  0.087*14, 0.087*15),
      HBHOCellParameters (15, 2, 1, 1, 5, 258.4,  HBRMAX, 0.087*14, 0.087*15),
      HBHOCellParameters (16, 1, 1, 1, 5, HBRMIN, 190.4,  0.087*15, 0.087*16),
      HBHOCellParameters (16, 2, 1, 1, 5, 190.4,  232.6,  0.087*15, 0.087*16)
    };
    int nCells = sizeof(cells)/sizeof(HBHOCellParameters);
    std::vector <HBHOCellParameters> result;
    result.reserve (nCells);
    for (int i = 0; i < nCells; ++i) result.push_back (cells[i]);
    return result;
  }

  // ----------> HO <-----------
  std::vector <HBHOCellParameters> makeHOCells () {
    const float HORMIN0 = 390.0;
    const float HORMIN1 = 412.6;
    const float HORMAX = 413.6;
    
    HBHOCellParameters cells [] = {
      // eta, depth, firstPhi, stepPhi, deltaPhi, rMin, rMax, etaMin, etaMax
      HBHOCellParameters ( 1, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*0, 0.087*1),
      HBHOCellParameters ( 2, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*1, 0.087*2),
      HBHOCellParameters ( 3, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*2, 0.087*3),
      HBHOCellParameters ( 4, 4, 1, 1, 5, HORMIN0, HORMAX, 0.087*3, 0.3075),
      HBHOCellParameters ( 5, 4, 1, 1, 5, HORMIN1, HORMAX, 0.3395,  0.087*5),
      HBHOCellParameters ( 6, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*5, 0.087*6),
      HBHOCellParameters ( 7, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*6, 0.087*7),
      HBHOCellParameters ( 8, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*7, 0.087*8),
      HBHOCellParameters ( 9, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*8, 0.087*9),
      HBHOCellParameters (10, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*9,  0.8494),
      HBHOCellParameters (11, 4, 1, 1, 5, HORMIN1, HORMAX, 0.873, 0.087*11),
      HBHOCellParameters (12, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*11, 0.087*12),
      HBHOCellParameters (13, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*12, 0.087*13),
      HBHOCellParameters (14, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*13, 0.087*14),
      HBHOCellParameters (15, 4, 1, 1, 5, HORMIN1, HORMAX, 0.087*14, 0.087*15)
    };
    int nCells = sizeof(cells)/sizeof(HBHOCellParameters);
    std::vector <HBHOCellParameters> result;
    result.reserve (nCells);
    for (int i = 0; i < nCells; ++i) result.push_back (cells[i]);
    return result;
  }

  // ----------> HE <-----------
  std::vector <HECellParameters> makeHECells () {
    const float HEZMIN = 400.458;
    const float HEZMID = 436.168;
    const float HEZMAX = 549.268;
    
    HECellParameters cells [] = {
      // eta, depth, firstPhi, stepPhi, deltaPhi, zMin, zMax, etaMin, etaMax
      HECellParameters ( 16, 3, 1, 1, 5,418.768,470.968, 0.087*15, 0.087*16),
      HECellParameters ( 17, 1, 1, 1, 5,409.698,514.468, 0.087*16, 0.087*17),
      HECellParameters ( 18, 1, 1, 1, 5,391.883,427.468, 0.087*17, 0.087*18),
      HECellParameters ( 18, 2, 1, 1, 5,427.468,540.568, 0.087*17, 0.087*18),
      HECellParameters ( 19, 1, 1, 1, 5, HEZMIN, HEZMID, 0.087*18, 0.087*19),
      HECellParameters ( 19, 2, 1, 1, 5, HEZMID, HEZMAX, 0.087*18, 0.087*19),
      HECellParameters ( 20, 1, 1, 1, 5, HEZMIN, HEZMID, 0.087*19, 1.74),
      HECellParameters ( 20, 2, 1, 1, 5, HEZMID, HEZMAX, 0.087*19, 1.74),
      HECellParameters ( 21, 1, 1, 2,10, HEZMIN, HEZMID, 1.74, 1.83),
      HECellParameters ( 21, 2, 1, 2,10, HEZMID, HEZMAX, 1.74, 1.83),
      HECellParameters ( 22, 1, 1, 2,10, HEZMIN, HEZMID, 1.83, 1.93),
      HECellParameters ( 22, 2, 1, 2,10, HEZMID, HEZMAX, 1.83, 1.93),
      HECellParameters ( 23, 1, 1, 2,10, HEZMIN, HEZMID, 1.93, 2.043),
      HECellParameters ( 23, 2, 1, 2,10, HEZMID, HEZMAX, 1.93, 2.043),
      HECellParameters ( 24, 1, 1, 2,10, HEZMIN, HEZMID, 2.043, 2.172),
      HECellParameters ( 24, 2, 1, 2,10, HEZMID, HEZMAX, 2.043, 2.172),
      HECellParameters ( 25, 1, 1, 2,10, HEZMIN, HEZMID, 2.172, 2.322),
      HECellParameters ( 25, 2, 1, 2,10, HEZMID, HEZMAX, 2.172, 2.322),
      HECellParameters ( 26, 1, 1, 2,10, HEZMIN, HEZMID, 2.322, 2.500),
      HECellParameters ( 26, 2, 1, 2,10, HEZMID, HEZMAX, 2.322, 2.500),
      HECellParameters ( 27, 1, 1, 2,10, HEZMIN,418.768, 2.500, 2.650),
      HECellParameters ( 27, 2, 1, 2,10,418.768, HEZMID, 2.500, 2.650),
      HECellParameters ( 27, 3, 1, 2,10, HEZMID, HEZMAX, 2.500, 2.650),
      HECellParameters ( 28, 1, 1, 2,10, HEZMIN,418.768, 2.650, 2.868),
      HECellParameters ( 28, 2, 1, 2,10,418.768, HEZMID, 2.650, 2.868),
      HECellParameters ( 28, 3, 1, 2,10, HEZMID, HEZMAX, 2.650, 3.000),
      HECellParameters ( 29, 1, 1, 2,10, HEZMIN, HEZMID, 2.868, 3.000),
      HECellParameters ( 29, 2, 1, 2,10,418.768, HEZMID, 2.868, 3.000)
    };
    int nCells = sizeof(cells)/sizeof(HECellParameters);
    std::vector <HECellParameters> result;
    result.reserve (nCells);
    for (int i = 0; i < nCells; ++i) result.push_back (cells[i]);
    return result;
  }


  // ----------> HE @ H2 <-----------
  std::vector <HECellParameters> makeHECells_H2 () {
    const float HEZMIN_H2 = 400.715;
    const float HEZMID_H2 = 436.285;
    const float HEZMAX_H2 = 541.885;
    
    HECellParameters cells [] = {
      // eta, depth, firstPhi, stepPhi, deltaPhi, zMin, zMax, etaMin, etaMax
      HECellParameters ( 16, 3, 1, 1, 5, 409.885,   462.685,   1.305, 1.373),
      HECellParameters ( 17, 1, 1, 1, 5, HEZMIN_H2, 427.485,   1.373, 1.444),
      HECellParameters ( 17, 2, 1, 1, 5, 427.485,   506.685,   1.373, 1.444),
      HECellParameters ( 18, 1, 1, 1, 5, HEZMIN_H2, HEZMID_H2, 1.444, 1.521),
      HECellParameters ( 18, 2, 1, 1, 5, HEZMID_H2, 524.285,   1.444, 1.521),
      HECellParameters ( 19, 1, 1, 1, 5, HEZMIN_H2, HEZMID_H2, 1.521, 1.603),
      HECellParameters ( 19, 2, 1, 1, 5, HEZMID_H2, HEZMAX_H2, 1.521, 1.603),
      HECellParameters ( 20, 1, 1, 1, 5, HEZMIN_H2, HEZMID_H2, 1.603, 1.693),
      HECellParameters ( 20, 2, 1, 1, 5, HEZMID_H2, HEZMAX_H2, 1.603, 1.693),
      HECellParameters ( 21, 1, 1, 2, 5, HEZMIN_H2, HEZMID_H2, 1.693, 1.79),
      HECellParameters ( 21, 2, 1, 2, 5, HEZMID_H2, HEZMAX_H2, 1.693, 1.79),
      HECellParameters ( 22, 1, 1, 2,10, HEZMIN_H2, HEZMID_H2, 1.79, 1.88),
      HECellParameters ( 22, 2, 1, 2,10, HEZMID_H2, HEZMAX_H2, 1.79, 1.88),
      HECellParameters ( 23, 1, 1, 2,10, HEZMIN_H2, HEZMID_H2, 1.88, 1.98),
      HECellParameters ( 23, 2, 1, 2,10, HEZMID_H2, HEZMAX_H2, 1.88, 1.98),
      HECellParameters ( 24, 1, 1, 2,10, HEZMIN_H2, 418.685,   1.98, 2.09),
      HECellParameters ( 24, 2, 1, 2,10, 418.685,   HEZMID_H2, 1.98, 2.09),
      HECellParameters ( 24, 3, 1, 2,10, HEZMID_H2, HEZMAX_H2, 1.98, 2.09),
      HECellParameters ( 25, 1, 1, 2,10, HEZMIN_H2, 418.685,   2.09, 2.21),
      HECellParameters ( 25, 2, 1, 2,10, 418.685,   HEZMID_H2, 2.09, 2.21),
      HECellParameters ( 25, 3, 1, 2,10, HEZMID_H2, HEZMAX_H2, 2.09, 2.21)
    };
    int nCells = sizeof(cells)/sizeof(HECellParameters);
    std::vector <HECellParameters> result;
    result.reserve (nCells);
    for (int i = 0; i < nCells; ++i) result.push_back (cells[i]);
    return result;
  }

  // ----------> HF <-----------
  std::vector <HFCellParameters> makeHFCells () {
    const float HFZMIN1 = 1115.;
    const float HFZMIN2 = 1137.;
    const float HFZMAX = 1280.1;
    
    HFCellParameters cells [] = {
      // eta, depth, firstPhi, stepPhi, deltaPhi, zMin, zMax, rMin, rMax
      HFCellParameters (29, 1, 1, 2, 10, HFZMIN1, HFZMAX,116.2,130.0),
      HFCellParameters (29, 2, 1, 2, 10, HFZMIN2, HFZMAX,116.2,130.0),
      HFCellParameters (30, 1, 1, 2, 10, HFZMIN1, HFZMAX, 97.5,116.2),
      HFCellParameters (30, 2, 1, 2, 10, HFZMIN2, HFZMAX, 97.5,116.2),
      HFCellParameters (31, 1, 1, 2, 10, HFZMIN1, HFZMAX, 81.8, 97.5),
      HFCellParameters (31, 2, 1, 2, 10, HFZMIN2, HFZMAX, 81.8, 97.5),
      HFCellParameters (32, 1, 1, 2, 10, HFZMIN1, HFZMAX, 68.6, 81.8),
      HFCellParameters (32, 2, 1, 2, 10, HFZMIN2, HFZMAX, 68.6, 81.8),
      HFCellParameters (33, 1, 1, 2, 10, HFZMIN1, HFZMAX, 57.6, 68.6),
      HFCellParameters (33, 2, 1, 2, 10, HFZMIN2, HFZMAX, 57.6, 68.6),
      HFCellParameters (34, 1, 1, 2, 10, HFZMIN1, HFZMAX, 48.3, 57.6),
      HFCellParameters (34, 2, 1, 2, 10, HFZMIN2, HFZMAX, 48.3, 57.6),
      HFCellParameters (35, 1, 1, 2, 10, HFZMIN1, HFZMAX, 40.6, 48.3),
      HFCellParameters (35, 2, 1, 2, 10, HFZMIN2, HFZMAX, 40.6, 48.3),
      HFCellParameters (36, 1, 1, 2, 10, HFZMIN1, HFZMAX, 34.0, 40.6),
      HFCellParameters (36, 2, 1, 2, 10, HFZMIN2, HFZMAX, 34.0, 40.6),
      HFCellParameters (37, 1, 1, 2, 10, HFZMIN1, HFZMAX, 28.6, 34.0),
      HFCellParameters (37, 2, 1, 2, 10, HFZMIN2, HFZMAX, 28.6, 34.0),
      HFCellParameters (38, 1, 1, 2, 10, HFZMIN1, HFZMAX, 24.0, 28.6),
      HFCellParameters (38, 2, 1, 2, 10, HFZMIN2, HFZMAX, 24.0, 28.6),
      HFCellParameters (39, 1, 1, 2, 10, HFZMIN1, HFZMAX, 20.1, 24.0),
      HFCellParameters (39, 2, 1, 2, 10, HFZMIN2, HFZMAX, 20.1, 24.0),
      HFCellParameters (40, 1, 3, 4, 20, HFZMIN1, HFZMAX, 16.9, 20.1),
      HFCellParameters (40, 2, 3, 4, 20, HFZMIN2, HFZMAX, 16.9, 20.1),
      HFCellParameters (41, 1, 3, 4, 20, HFZMIN1, HFZMAX, 12.5, 16.9),
      HFCellParameters (41, 2, 3, 4, 20, HFZMIN2, HFZMAX, 12.5, 16.9)
    };
    int nCells = sizeof(cells)/sizeof(HFCellParameters);
    std::vector <HFCellParameters> result;
    result.reserve (nCells);
    for (int i = 0; i < nCells; ++i) result.push_back (cells[i]);
    return result;
  }

  //
  // Convert constants to appropriate cells
  //
  void fillHBHO (CaloSubdetectorGeometry* fGeometry, const std::vector <HBHOCellParameters>& fCells, bool fHB) {
    for (size_t iCell = 0; iCell < fCells.size(); ++iCell) {
      const HBHOCellParameters& param = fCells[iCell];
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
	  cellParams.push_back (0.5 * (param.etaMax - param.etaMin)); // deta_half
	  cellParams.push_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	  cellParams.push_back (0.5 * (param.rMax - param.rMin) * cosh (etaCenter)); // dr_half
	  cellParams.push_back ( fabs( refPoint.eta()));
	  cellParams.push_back ( fabs( refPoint.z() ) ) ;
	  
// 	  std::cout << "HcalFlexiHardcodeGeometryLoader::fillHBHO-> " << hid << hid.ieta() << '/' << hid.iphi() << '/' << hid.depth()
// 		    << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
	  
	  fGeometry->newCell( refPoint,  refPoint,  refPoint, 
			      CaloCellGeometry::getParmPtr(
				 cellParams, 
				 fGeometry->parMgr(), 
				 fGeometry->parVecVec() ),
			      hid ) ;
	}
      }
    }
  }
  
  void fillHE (CaloSubdetectorGeometry* fGeometry, const std::vector <HECellParameters>& fCells) {
    for (size_t iCell = 0; iCell < fCells.size(); ++iCell) {
      const HECellParameters& param = fCells[iCell];
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
	  cellParams.push_back (0.5 * (param.etaMax - param.etaMin)); // deta_half
	  cellParams.push_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	  cellParams.push_back (-0.5 * (param.zMax - param.zMin) / tanh (etaCenter)); // dz_half, "-" means edges in Z
	  cellParams.push_back ( fabs( refPoint.eta()));
	  cellParams.push_back ( fabs( refPoint.z() ) ) ;
	  
// 	  std::cout << "HcalFlexiHardcodeGeometryLoader::fillHE-> " << hid << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
	  
	  fGeometry->newCell( refPoint,  refPoint,  refPoint, 
			      CaloCellGeometry::getParmPtr(
				 cellParams, 
				 fGeometry->parMgr(), 
				 fGeometry->parVecVec() ) ,
			      hid ) ;
	}
      }
    }
  }

  void fillHF (CaloSubdetectorGeometry* fGeometry, const std::vector <HFCellParameters>& fCells) {
    for (size_t iCell = 0; iCell < fCells.size(); ++iCell) {
      const HFCellParameters& param = fCells[iCell];
      for (int iPhi = param.phiFirst; iPhi <= MAX_HCAL_PHI; iPhi += param.phiStep) {
	for (int iside = -1; iside <= 1; iside += 2) { // both detector sides are identical
	  HcalDetId hid (HcalForward, param.eta*iside, iPhi, param.depth);
	  float phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD; // middle of the cell
	  GlobalPoint inner (param.rMin, 0., param.zMin);
	  GlobalPoint outer (param.rMax, 0., param.zMin);
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
	  
// 	  std::cout << "HcalFlexiHardcodeGeometryLoader::fillHF-> " << hid << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
	  
	  fGeometry->newCell( refPoint,  refPoint,  refPoint, 
			      CaloCellGeometry::getParmPtr(
				 cellParams, 
				 fGeometry->parMgr(), 
				 fGeometry->parVecVec() ),
			      hid ) ;
	}
      }
    }
  }

}  // end of local stuff

// ==============> Loader Itself <==========================

HcalFlexiHardcodeGeometryLoader::HcalFlexiHardcodeGeometryLoader()
{
}

CaloSubdetectorGeometry* HcalFlexiHardcodeGeometryLoader::load(const HcalTopology& fTopology) {
  CaloSubdetectorGeometry* hcalGeometry = new HcalGeometry (&fTopology);
  if( 0 == hcalGeometry->cornersMgr() ) hcalGeometry->allocateCorners ( 
     HcalGeometry::k_NumberOfCellsForCorners ) ;
  if( 0 == hcalGeometry->parMgr() ) hcalGeometry->allocatePar (
     HcalGeometry::k_NumberOfParametersPerShape*HcalGeometry::k_NumberOfShapes,
     HcalGeometry::k_NumberOfParametersPerShape ) ;
  // ugly kluge to extract H2 mode from the topology 
  if (fTopology.firstHEDoublePhiRing() < 22) { // regular geometry
    fillHBHO (hcalGeometry, makeHBCells(), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHF (hcalGeometry, makeHFCells());
    fillHE (hcalGeometry, makeHECells());
  }
  else { // TB geometry
    fillHBHO (hcalGeometry, makeHBCells(), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHE (hcalGeometry, makeHECells_H2());
  }
  return hcalGeometry;
}



