#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <algorithm>

typedef CaloCellGeometry::CCGFloat CCGFloat ;

//#define EDM_ML_DEBUG

// ==============> Loader Itself <==========================

HcalFlexiHardcodeGeometryLoader::HcalFlexiHardcodeGeometryLoader(const edm::ParameterSet&) {

  MAX_HCAL_PHI = 72;
  DEGREE2RAD = M_PI / 180.;
}

CaloSubdetectorGeometry* HcalFlexiHardcodeGeometryLoader::load(const HcalTopology& fTopology, const HcalDDDRecConstants& hcons) {
  HcalGeometry* hcalGeometry = new HcalGeometry (fTopology);
  if( nullptr == hcalGeometry->cornersMgr() ) hcalGeometry->allocateCorners ( fTopology.ncells()+fTopology.getHFSize() );
  if( nullptr == hcalGeometry->parMgr() ) hcalGeometry->allocatePar (hcalGeometry->numberOfShapes(),
							       HcalGeometry::k_NumberOfParametersPerShape ) ;
  isBH_ = hcons.isBH();
#ifdef EDM_ML_DEBUG
  std::cout << "FlexiGeometryLoader initialize with ncells " 
	    << fTopology.ncells() << " and shapes " 
	    << hcalGeometry->numberOfShapes() << ":"
	    << HcalGeometry::k_NumberOfParametersPerShape 
	    << " with BH Flag " << isBH_ << std::endl;
#endif
  if (fTopology.mode() == HcalTopologyMode::H2) {  // TB geometry
    fillHBHO (hcalGeometry, makeHBCells(hcons), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHE (hcalGeometry, makeHECells_H2());
  } else { // regular geometry
    fillHBHO (hcalGeometry, makeHBCells(hcons), true);
    fillHBHO (hcalGeometry, makeHOCells(), false);
    fillHF (hcalGeometry, makeHFCells(hcons));
    fillHE (hcalGeometry, makeHECells(hcons));
  }
  //fast insertion of valid ids requires sort at end
  hcalGeometry->sortValidIds();
  return hcalGeometry;
}


// ----------> HB <-----------
std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters> HcalFlexiHardcodeGeometryLoader::makeHBCells (const HcalDDDRecConstants& hcons) {

  std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters> result;
  std::vector<std::pair<double,double> > gconsHB = hcons.getConstHBHE(0);
  std::vector<HcalDDDRecConstants::HcalEtaBin> etabins = hcons.getEtaBins(0);

#ifdef EDM_ML_DEBUG
  std::cout << "FlexiGeometryLoader called for " << etabins.size() 
	    << " Eta Bins" << std::endl;
  for (unsigned int k=0; k<gconsHB.size(); ++k) {
    std::cout << "gconsHB[" << k << "] = " << gconsHB[k].first << " +- "
	      << gconsHB[k].second << std::endl;
  }
#endif
  for (auto & etabin : etabins) {
    int iring = (etabin.zside >= 0) ? etabin.ieta : -etabin.ieta;
    int depth = etabin.depthStart;
    double dphi = (etabin.phis.size() > 1) ? 
      (etabin.phis[1].second-etabin.phis[0].second) : 
      ((2.0*M_PI)/MAX_HCAL_PHI);
    for (unsigned int k=0; k<etabin.layer.size(); ++k) {
      int layf = etabin.layer[k].first-1;
      int layl = etabin.layer[k].second-1;
      double rmin = gconsHB[layf].first-gconsHB[layf].second;
      double rmax = gconsHB[layl].first+gconsHB[layl].second;
      for (unsigned int j=0; j<etabin.phis.size(); ++j) {
#ifdef EDM_ML_DEBUG
	std::cout << "HBRing " << iring << " eta " << etabins[i].etaMin << ":"
		  << etabins[i].etaMax << " depth " << depth << " R " << rmin
		  << ":" << rmax << " Phi " << etabins[i].phis[j].first << ":" 
		  << etabins[i].phis[j].second << ":" << dphi << " layer[" << k 
		  << "]: " << etabins[i].layer[k].first-1 << ":"
		  << etabins[i].layer[k].second << std::endl;
#endif
	result.emplace_back (HcalFlexiHardcodeGeometryLoader::HBHOCellParameters(iring, depth, etabin.phis[j].first, etabin.phis[j].second, dphi, rmin, rmax, etabin.etaMin, etabin.etaMax));
      }
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
  const int    nPhi    = 72;
  const double etamin[nCells] = {0.000,0.087,0.174, 0.261, 0.3395,0.435,0.522,
				 0.609,0.696,0.783, 0.873, 0.957, 1.044,1.131,
				 1.218};
  const double etamax[nCells] = {0.087,0.174,0.261, 0.3075,0.435, 0.522,0.609,
				 0.696,0.783,0.8494,0.957, 1.044, 1.131,1.218,
				 1.305};
  std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters> result;
  result.reserve (nCells*nPhi);
  double dphi = ((2.0*M_PI)/nPhi);
  for (int i = 0; i < nCells; ++i) {
    double rmin = ((i < 4) ? HORMIN0 : HORMIN1);
    for (int iside = -1; iside <= 1; iside += 2) {
      for (int j=0; j < nPhi; ++j) {
	double phi = (j+0.5)*dphi;
	// eta, depth, phi, phi0, deltaPhi, rMin, rMax, etaMin, etaMax
	result.emplace_back (HcalFlexiHardcodeGeometryLoader::HBHOCellParameters(iside*(i+1), 4, j+1, phi, dphi, rmin, HORMAX, etamin[i], etamax[i]));
      }
    }
  }
  return result;
}


//
// Convert constants to appropriate cells
//
void HcalFlexiHardcodeGeometryLoader::fillHBHO (HcalGeometry* fGeometry, const std::vector<HcalFlexiHardcodeGeometryLoader::HBHOCellParameters>& fCells, bool fHB) {

  fGeometry->increaseReserve(fCells.size());
  for (const auto & param : fCells) {
    HcalDetId hid (fHB ? HcalBarrel : HcalOuter, param.ieta, param.iphi, param.depth);
    float phiCenter = param.phi; // middle of the cell
    float etaCenter = 0.5*(param.etaMin + param.etaMax);
    float x = param.rMin* cos (phiCenter);
    float y = param.rMin* sin (phiCenter);
    float z = (param.ieta < 0) ? -(param.rMin*sinh(etaCenter)) : (param.rMin*sinh(etaCenter));
    // make cell geometry
    GlobalPoint refPoint (x,y,z); // center of the cell's face
    std::vector<CCGFloat> cellParams;
    cellParams.reserve (5);
    cellParams.emplace_back (0.5 * (param.etaMax - param.etaMin)); // deta_half
    cellParams.emplace_back (0.5 * param.dphi);                    // dphi_half
    cellParams.emplace_back (0.5 * (param.rMax - param.rMin) * cosh (etaCenter)); // dr_half
    cellParams.emplace_back ( fabs( refPoint.eta() ) ) ;
    cellParams.emplace_back ( fabs( refPoint.z() ) ) ;
#ifdef EDM_ML_DEBUG
    std::cout << "HcalFlexiHardcodeGeometryLoader::fillHBHO-> " << hid 
	      << " " << hid.rawId() << " " << std::hex << hid.rawId() 
	      << std::dec << " " << hid << " " << refPoint << '/' 
	      << cellParams [0] << '/' << cellParams [1] << '/' 
	      << cellParams [2] << std::endl;
#endif
    fGeometry->newCellFast(refPoint,  refPoint,  refPoint, 
		       CaloCellGeometry::getParmPtr(cellParams, 
						    fGeometry->parMgr(), 
						    fGeometry->parVecVec()),
		       hid ) ;
  }
}


// ----------> HE <-----------
std::vector<HcalFlexiHardcodeGeometryLoader::HECellParameters> HcalFlexiHardcodeGeometryLoader::makeHECells (const HcalDDDRecConstants& hcons) {

  std::vector<HcalFlexiHardcodeGeometryLoader::HECellParameters> result;
  std::vector<std::pair<double,double> > gconsHE = hcons.getConstHBHE(1);
#ifdef EDM_ML_DEBUG
  std::cout << "HcalFlexiHardcodeGeometryLoader:HE with " << gconsHE.size()
	    << " cells" << std::endl;
#endif
  if (!gconsHE.empty()) {
    std::vector<HcalDDDRecConstants::HcalEtaBin> etabins = hcons.getEtaBins(1);

#ifdef EDM_ML_DEBUG
    std::cout << "FlexiGeometryLoader called for HE with " << etabins.size() 
	      << " Eta Bins and " << gconsHE.size() << " depths" 
	      << std::endl;
    for (unsigned int i=0; i<gconsHE.size(); ++i)
      std::cout << " Depth[" << i << "] = " << gconsHE[i].first << " +- " 
		<< gconsHE[i].second;
    std::cout << std::endl;
#endif
    for (auto & etabin : etabins) {
      int    iring = (etabin.zside >= 0) ? etabin.ieta : -etabin.ieta;
      int    depth = etabin.depthStart;
      double dphi = (etabin.phis.size() > 1) ? 
	(etabin.phis[1].second-etabin.phis[0].second) : 
	((2.0*M_PI)/MAX_HCAL_PHI);
#ifdef EDM_ML_DEBUG
      std::cout << "FlexiGeometryLoader::Ring " << iring << " nphi " << nphi
		<< " dstart " << depth << " dphi " << dphi << " units "
		<< units << " fioff " << fioff << " layers "
		<< etabins[i].layer.size() << std::endl;
#endif
      for (unsigned int k=0; k<etabin.layer.size(); ++k) {
	int layf = etabin.layer[k].first-1;
	int layl = etabin.layer[k].second-1;
	double zmin = gconsHE[layf].first-gconsHE[layf].second;
	double zmax = gconsHE[layl].first+gconsHE[layl].second;
	if (zmin < 1.0) {
	  for (int k2=layf; k2<=layl; ++k2) {
	    if (gconsHE[k2].first > 10) {
	      zmin = gconsHE[k2].first-gconsHE[k2].second;
	      break;
	    }
	  }
	}
	if (zmin >= zmax) zmax = zmin+10.;
	for (unsigned int j=0; j<etabin.phis.size(); ++j) {
#ifdef EDM_ML_DEBUG
	  std::cout << "HERing " << iring << " eta " << etabins[i].etaMin << ":"
		    << etabins[i].etaMax << " depth " << depth << " Z " << zmin
		    << ":" << zmax << " Phi :" << etabins[i].phis[j].first 
		    << ":" << etabins[i].phis[j].second << ":" << dphi 
		    << " layer[" << k << "]: " << etabins[i].layer[k].first-1 
		    << ":" << etabins[i].layer[k].second-1 << std::endl;
#endif
	  result.emplace_back(HcalFlexiHardcodeGeometryLoader::HECellParameters(iring, depth, etabin.phis[j].first, etabin.phis[j].second, dphi, zmin, zmax, etabin.etaMin, etabin.etaMax));
	}
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
  const int    nEtas = 10;
  const int    nDepth[nEtas] = {1,2,2,2,2,2,2,2,3,3};
  const int    dStart[nEtas] = {3,1,1,1,1,1,1,1,1,1};
  const int    nPhis[nEtas]  = {8,8,8,8,8,8,4,4,4,4};
  const double etas[nEtas+1] = {1.305,1.373,1.444,1.521,1.603,1.693,1.790,
				1.880,1.980,2.090,2.210};
  const double zval[4*nEtas] = {409.885,462.685,0.,0.,
				HEZMIN_H2,427.485,506.685,0.0,
				HEZMIN_H2,HEZMID_H2,524.285,0.,
				HEZMIN_H2,HEZMID_H2,HEZMAX_H2,0.,
				HEZMIN_H2,HEZMID_H2,HEZMAX_H2,0.,
				HEZMIN_H2,HEZMID_H2,HEZMAX_H2,0.,
				HEZMIN_H2,HEZMID_H2,HEZMAX_H2,0.,
				HEZMIN_H2,HEZMID_H2,HEZMAX_H2,0.,
				HEZMIN_H2,418.685,HEZMID_H2,HEZMAX_H2,
				HEZMIN_H2,418.685,HEZMID_H2,HEZMAX_H2};
  std::vector<HcalFlexiHardcodeGeometryLoader::HECellParameters> result;

  for (int i = 0; i < nEtas; ++i) {
    int ieta = 16+i;
    for (int k=0; k<nDepth[i]; ++k) {
      int depth = dStart[i]+k;
      for (int j=0; j < nPhis[i]; ++j) {
	int    iphi = (nPhis[i] == 8) ? (j+1) : (2*j+1);
	double dphi = (40.0*DEGREE2RAD)/nPhis[i];
	double phi0 = (j+0.5)*dphi;
	// ieta, depth, iphi, phi0, deltaPhi, zMin, zMax, etaMin, etaMax
	result.emplace_back (HcalFlexiHardcodeGeometryLoader::HECellParameters(ieta, depth, iphi, phi0, dphi, zval[4*i+k+1], zval[4*i+k+2], etas[i], etas[i+1]));
      }
    }
  }
  return result;
}

// ----------> HF <-----------
std::vector <HcalFlexiHardcodeGeometryLoader::HFCellParameters> HcalFlexiHardcodeGeometryLoader::makeHFCells (const HcalDDDRecConstants& hcons) {

  const float HFZMIN1 = 1115.;
  const float HFZMIN2 = 1137.;
  const float HFZMAX = 1280.1;
  std::vector<HcalDDDRecConstants::HFCellParameters> cells = hcons.getHFCellParameters();
  unsigned int nCells = cells.size();
  std::vector <HcalFlexiHardcodeGeometryLoader::HFCellParameters> result;
  result.reserve (nCells);
  for (unsigned int i = 0; i < nCells; ++i) {
    HcalFlexiHardcodeGeometryLoader::HFCellParameters cell1(cells[i].ieta,cells[i].depth,cells[i].firstPhi,cells[i].stepPhi,cells[i].nPhi,5*cells[i].stepPhi,HFZMIN1,HFZMAX,cells[i].rMin,cells[i].rMax);
    result.emplace_back (cell1);
    HcalFlexiHardcodeGeometryLoader::HFCellParameters cell2(cells[i].ieta,1+cells[i].depth,cells[i].firstPhi,cells[i].stepPhi,cells[i].nPhi,5*cells[i].stepPhi,HFZMIN2,HFZMAX,cells[i].rMin,cells[i].rMax);
    result.emplace_back (cell2);
  }
  return result;
}
  
void HcalFlexiHardcodeGeometryLoader::fillHE (HcalGeometry* fGeometry, const std::vector <HcalFlexiHardcodeGeometryLoader::HECellParameters>& fCells) {

  fGeometry->increaseReserve(fCells.size());
  for (const auto & param : fCells) {
    HcalDetId hid (HcalEndcap, param.ieta, param.iphi, param.depth);
    float phiCenter = param.phi; // middle of the cell
    float etaCenter = 0.5 * (param.etaMin + param.etaMax);
    int   iside     = (param.ieta >= 0) ? 1 : -1;
    float perp = param.zMin / sinh (etaCenter);
    float x = perp * cos (phiCenter);
    float y = perp * sin (phiCenter);
    float z = (isBH_) ? (iside*0.5*(param.zMin+param.zMax)) : (iside*param.zMin);
    // make cell geometry
    GlobalPoint refPoint (x,y,z); // center of the cell's face
    std::vector<CCGFloat> cellParams;
    cellParams.reserve (5);
    cellParams.emplace_back (0.5 * (param.etaMax - param.etaMin)); //deta_half
    cellParams.emplace_back (0.5 * param.dphi);  // dphi_half
    cellParams.emplace_back (-0.5 * (param.zMax - param.zMin) / tanh (etaCenter)); // dz_half, "-" means edges in Z
    cellParams.emplace_back ( fabs( refPoint.eta() ) ) ;
    cellParams.emplace_back ( fabs( refPoint.z() ) ) ;
#ifdef EDM_ML_DEBUG
    std::cout << "HcalFlexiHardcodeGeometryLoader::fillHE-> " << hid << " "
	      << hid.rawId() << " " << std::hex << hid.rawId() << std::dec
	      << " " << hid << refPoint << '/' << cellParams [0] << '/' 
	      << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif
    fGeometry->newCellFast(refPoint,  refPoint,  refPoint, 
		       CaloCellGeometry::getParmPtr(cellParams, 
						    fGeometry->parMgr(), 
						    fGeometry->parVecVec()),
		       hid ) ;
  }
}

void HcalFlexiHardcodeGeometryLoader::fillHF (HcalGeometry* fGeometry, const std::vector <HcalFlexiHardcodeGeometryLoader::HFCellParameters>& fCells) {

  fGeometry->increaseReserve(fCells.size());
  for (const auto & param : fCells) {
    for (int kPhi = 0; kPhi < param.nPhi; ++kPhi) {
      int iPhi = param.phiFirst + kPhi*param.phiStep;
      HcalDetId hid (HcalForward, param.eta, iPhi, param.depth);
      // middle of the cell
      float phiCenter = ((iPhi-1)*360./MAX_HCAL_PHI + 0.5*param.dphi) * DEGREE2RAD;
      GlobalPoint inner (param.rMin, 0, param.zMin);
      GlobalPoint outer (param.rMax, 0, param.zMin);
      float iEta = inner.eta();
      float oEta = outer.eta();
      float etaCenter = 0.5 * ( iEta + oEta );
	
      float perp = param.zMin / sinh (etaCenter);
      float x = perp * cos (phiCenter);
      float y = perp * sin (phiCenter);
      float z = (param.eta > 0) ? param.zMin : -param.zMin;
      // make cell geometry
      GlobalPoint refPoint (x,y,z); // center of the cell's face
      std::vector<CCGFloat> cellParams;
      cellParams.reserve (5);
      cellParams.emplace_back (0.5 * ( iEta - oEta )); // deta_half
      cellParams.emplace_back (0.5 * param.dphi * DEGREE2RAD);  // dphi_half
      cellParams.emplace_back (0.5 * (param.zMax - param.zMin)); // dz_half
      cellParams.emplace_back ( fabs( refPoint.eta()));
      cellParams.emplace_back ( fabs( refPoint.z() ) ) ;
#ifdef EDM_ML_DEBUG
      std::cout << "HcalFlexiHardcodeGeometryLoader::fillHF-> " << hid << " " 
		<< hid.rawId() << " " << std::hex << hid.rawId() << std::dec 
		<< " " << hid << " " << refPoint << '/' << cellParams [0] 
		<< '/' << cellParams [1] << '/' << cellParams [2] << std::endl;
#endif	
      fGeometry->newCellFast(refPoint,  refPoint,  refPoint, 
			 CaloCellGeometry::getParmPtr(cellParams, 
						      fGeometry->parMgr(), 
						      fGeometry->parVecVec()),
			 hid ) ;
    }
  }
}
