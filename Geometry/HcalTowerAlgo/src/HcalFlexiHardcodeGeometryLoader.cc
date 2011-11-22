#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

namespace {
  const int MAX_HCAL_PHI = 72;
  const double DEGREE2RAD = M_PI / 180.;

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
  std::vector <HBHOCellParameters> makeHBCells (const HcalTopology & topology) {
    const float HBRMIN = 181.1;
    const float HBRMAX = 288.8;
    
    float normalDepths[] = {HBRMIN, HBRMAX};
    float ring15Depths[] = {HBRMIN, 258.4, HBRMAX};
    float ring16Depths[] = {HBRMIN, 190.4, 232.6};
    float slhcDepths[] = {HBRMIN, 214., 239., HBRMAX};
    std::vector <HBHOCellParameters> result;
    for(int iring = 1; iring <= 16; ++iring)
    {
      float * depths = slhcDepths;
      if(topology.mode() != HcalTopology::md_SLHC)
      {
        if(iring == 15) depths = ring15Depths;
        else if(iring == 16) depths = ring16Depths;
        else depths = normalDepths;
      }

      int ndepth, startingDepth;
      topology.depthBinInformation(HcalBarrel, iring, ndepth, startingDepth);
      for(int idepth = startingDepth; idepth <= ndepth; ++idepth)
      {
        float rmin = depths[idepth-1];
        float rmax = depths[idepth];
        result.push_back(HBHOCellParameters(iring, idepth, 1, 1, 5, rmin, rmax, (iring-1)*0.087, iring*0.087));
      }
    }
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
  std::vector <HECellParameters> makeHECells (const HcalTopology & topology) {
    std::vector <HECellParameters> result;
    const float HEZMIN = 400.458;
    const float HEZMID = 436.168;
    const float HEZMAX = 549.268;
    float normalDepths[] = {HEZMIN, HEZMID, HEZMAX};
    float tripleDepths[] = {HEZMIN, 418.768, HEZMID, HEZMAX};
    float slhcDepths[] = {HEZMIN, 418.768, HEZMID, 493., HEZMAX};
    float ring16Depths[] = {418.768,470.968};
    float ring16slhcDepths[] = {418.768, 450., 470.968};
    float ring17Depths[] = {409.698,514.468};
    float ring17slhcDepths[] = {409.698, 435., 460., 495., 514.468};
    float ring18Depths[] = {391.883,427.468,540.568};
    float ring18slhcDepths[] = {391.883, 439.,  467., 504. , 540.568};
    float etaBounds[] = {0.087*15, 0.087*16, 0.087*17, 0.087*18,  0.087*19,
                         1.74, 1.83,  1.93, 2.043, 2.172, 2.322, 2.500,
                         2.650, 2.868, 3.000};

    // count by ring - 16
    bool slhc = (topology.mode() == HcalTopology::md_SLHC);
    for(int iringm16=0; iringm16 <= 13; ++iringm16)
    {
      int iring = iringm16 + 16;
      float * depths = slhcDepths;
      if(iring == 16) depths = (slhc? ring16slhcDepths : ring16Depths);
      else if(iring == 17) depths = (slhc ? ring17slhcDepths : ring17Depths);
      else if(iring == 18) depths = (slhc ? ring18slhcDepths : ring18Depths);
      else if(!slhc) depths = (iring >= topology.firstHETripleDepthRing() ? tripleDepths : normalDepths);
      float etamin = etaBounds[iringm16];
      float etamax = etaBounds[iringm16+1];
      int ndepth, startingDepth;
      topology.depthBinInformation(HcalEndcap, iring, ndepth, startingDepth);
      for(int idepth = 0; idepth < ndepth; ++idepth)
      {
        int depthIndex = idepth + startingDepth;
        float zmin = depths[idepth];
        float zmax = depths[idepth+1];
        int stepPhi = (iring >= topology.firstHEDoublePhiRing() ? 2 : 1);
        int deltaPhi =  (iring >= topology.firstHEDoublePhiRing() ? 10 : 5);
        result.push_back(HECellParameters(iring, depthIndex, 1, stepPhi, deltaPhi, zmin, zmax, etamin, etamax));
      }
    }

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
	  double phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD; // middle of the cell
	  double etaCenter = 0.5*(param.etaMin + param.etaMax);
	  double x = param.rMin* cos (phiCenter);
	  double y = param.rMin* sin (phiCenter);
	  double z = iside * param.rMin * sinh(etaCenter);
	  // make cell geometry
	  GlobalPoint refPoint (x,y,z); // center of the cell's face
	  std::vector<double> cellParams;
	  cellParams.reserve (5);
	  cellParams.push_back (0.5 * (param.etaMax - param.etaMin)); // deta_half
	  cellParams.push_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	  cellParams.push_back (0.5 * (param.rMax - param.rMin) * cosh (etaCenter)); // dr_half
	  cellParams.push_back ( fabs( refPoint.eta() ) ) ;
	  cellParams.push_back ( fabs( refPoint.z() ) ) ;
	  
// 	  std::cout << "HcalFlexiHardcodeGeometryLoader::fillHBHO-> " << hid << hid.ieta() << '/' << hid.iphi() << '/' << hid.depth()
// 		    << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
	  
	  CaloCellGeometry* newcell = 
	     new calogeom::IdealObliquePrism( refPoint, 
					      fGeometry->cornersMgr(),
					      CaloCellGeometry::getParmPtr(
						 cellParams, 
						 fGeometry->parMgr(), 
						 fGeometry->parVecVec()));
	  // ... and store it
	  fGeometry->addCell (hid, newcell);						       
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
	  double phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD; // middle of the cell
	  double etaCenter = 0.5 * (param.etaMin + param.etaMax);

	  double perp = param.zMin / sinh (etaCenter);
	  double x = perp * cos (phiCenter);
	  double y = perp * sin (phiCenter);
	  double z = iside * param.zMin;
	  // make cell geometry
	  GlobalPoint refPoint (x,y,z); // center of the cell's face
	  std::vector<double> cellParams;
	  cellParams.reserve (5);
	  cellParams.push_back (0.5 * (param.etaMax - param.etaMin)); // deta_half
	  cellParams.push_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	  cellParams.push_back (-0.5 * (param.zMax - param.zMin) / tanh (etaCenter)); // dz_half, "-" means edges in Z
	  cellParams.push_back ( fabs( refPoint.eta() ) ) ;
	  cellParams.push_back ( fabs( refPoint.z() ) ) ;
	  
// 	  std::cout << "HcalFlexiHardcodeGeometryLoader::fillHE-> " << hid << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
	  
	  CaloCellGeometry* newcell = 
	    new calogeom::IdealObliquePrism( refPoint, 
					     fGeometry->cornersMgr(),
					     CaloCellGeometry::getParmPtr(
						cellParams, 
						fGeometry->parMgr(), 
						fGeometry->parVecVec()));
	  // ... and store it
	  fGeometry->addCell (hid, newcell);						       
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
	  double phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD; // middle of the cell
	  GlobalPoint inner (param.rMin, 0, param.zMin);
	  GlobalPoint outer (param.rMax, 0, param.zMin);
	  double etaCenter = 0.5 * (inner.eta() + outer.eta());

	  double perp = param.zMin / sinh (etaCenter);
	  double x = perp * cos (phiCenter);
	  double y = perp * sin (phiCenter);
	  double z = iside * param.zMin;
	  // make cell geometry
	  GlobalPoint refPoint (x,y,z); // center of the cell's face
	  std::vector<double> cellParams; cellParams.reserve (3);
	  cellParams.push_back (0.5 * (inner.eta() - outer.eta())); // deta_half
	  cellParams.push_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
	  cellParams.push_back (0.5 * (param.zMax - param.zMin)); // dz_half
	  cellParams.push_back ( fabs( refPoint.eta() ) ) ;
	  cellParams.push_back ( fabs( refPoint.z() ) ) ;
	  
// 	  std::cout << "HcalFlexiHardcodeGeometryLoader::fillHF-> " << hid << refPoint << '/' << cellParams [0] << '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
	  
	  CaloCellGeometry* newcell = 
	    new calogeom::IdealZPrism( refPoint, 
				       fGeometry->cornersMgr(),
				       CaloCellGeometry::getParmPtr(
					  cellParams, 
					  fGeometry->parMgr(), 
					  fGeometry->parVecVec()));
	  // ... and store it
	  fGeometry->addCell (hid, newcell);						       
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
  if (fTopology.mode() == HcalTopology::md_H2) {  // TB geometry
    fillHBHO (hcalGeometry, makeHBCells(fTopology), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHE (hcalGeometry, makeHECells_H2());
  }
  else { // regular geometry
    fillHBHO (hcalGeometry, makeHBCells(fTopology), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHF (hcalGeometry, makeHFCells());
    fillHE (hcalGeometry, makeHECells(fTopology));
  }
  return hcalGeometry;
}



