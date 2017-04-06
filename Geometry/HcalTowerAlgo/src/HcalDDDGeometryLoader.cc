#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometryLoader.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<string>

typedef CaloCellGeometry::CCGFloat CCGFloat ;

//#define EDM_ML_DEBUG

HcalDDDGeometryLoader::HcalDDDGeometryLoader(const HcalDDDRecConstants* hcons)
  : hcalConstants(hcons) { 
  isBH_ = hcalConstants->isBH();
}

HcalDDDGeometryLoader::~HcalDDDGeometryLoader() {
}

HcalDDDGeometryLoader::ReturnType 
HcalDDDGeometryLoader::load(const HcalTopology& topo, DetId::Detector det, int subdet) {

  HcalSubdetector  hsub        = static_cast<HcalSubdetector>(subdet);


  HcalDDDGeometry* gDDD ( new HcalDDDGeometry(topo) );
  ReturnType geom ( gDDD );

  if ( geom->cornersMgr() == 0 ) {
     const unsigned int count (hcalConstants->numberOfCells(HcalBarrel ) +
			       hcalConstants->numberOfCells(HcalEndcap ) +
			       2*hcalConstants->numberOfCells(HcalForward) +
			       hcalConstants->numberOfCells(HcalOuter  ) );
     geom->allocateCorners( count ) ;
  }

  //  if( geom->cornersMgr() == 0 )  geom->allocateCorners( 2592 ) ;

  if ( geom->parMgr()     == 0 ) geom->allocatePar( 500, 3 ) ;

  fill (hsub, gDDD, geom );
  return geom ;
}

HcalDDDGeometryLoader::ReturnType 
HcalDDDGeometryLoader::load(const HcalTopology& topo) {

  HcalDDDGeometry* gDDD ( new HcalDDDGeometry(topo) );
  ReturnType geom ( gDDD );

  if( geom->cornersMgr() == 0 ) {
    const unsigned int count (hcalConstants->numberOfCells(HcalBarrel ) +
			      hcalConstants->numberOfCells(HcalEndcap ) +
			      2*hcalConstants->numberOfCells(HcalForward) +
			      hcalConstants->numberOfCells(HcalOuter  ) );
    geom->allocateCorners( count ) ;
  }
  if( geom->parMgr()     == 0 ) geom->allocatePar( 500, 3 ) ;
  
  fill(HcalBarrel,  gDDD, geom); 
  fill(HcalEndcap,  gDDD, geom); 
  fill(HcalForward, gDDD, geom); 
  fill(HcalOuter,   gDDD, geom);
  return geom ;
}

void HcalDDDGeometryLoader::fill(HcalSubdetector          subdet, 
				 HcalDDDGeometry*         geometryDDD,
				 CaloSubdetectorGeometry* geom           ) {

  // start by making the new HcalDetIds
  std::vector<HcalCellType> hcalCells = hcalConstants->HcalCellTypes(subdet);
  geometryDDD->insertCell(hcalCells);
#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDGeometryLoader::fill gets " << hcalCells.size() 
	    << " cells for subdetector " << subdet << std::endl;
#endif			 
  // Make the new HcalDetIds and the cells

  std::vector<HcalDetId> hcalIds;
  for (unsigned int i=0; i<hcalCells.size(); i++) {
    int etaRing  = hcalCells[i].etaBin();
    int iside    = hcalCells[i].zside();
    int depthBin = hcalCells[i].depthSegment();
    double dphi  = hcalCells[i].phiBinWidth();
    std::vector<std::pair<int,double> > phis = hcalCells[i].phis();
#ifdef EDM_ML_DEBUG
    std::cout << "HcalDDDGeometryLoader: Subdet " << subdet << " side "
	      << iside << " eta " << etaRing << " depth " << depthBin 
	      << " with " << phis.size() << "modules:" << std::endl;
#endif
    for (unsigned int k = 0; k < phis.size(); k++) {
#ifdef EDM_ML_DEBUG
      std::cout << "HcalDDDGeometryLoader::fill Cell " << i << " eta " 
		<< iside*etaRing << " phi " << phis[k].first << "("
		<< phis[k].second/CLHEP::deg << ", " << dphi/CLHEP::deg 
		<< ") depth " << depthBin << std::endl;
#endif
      HcalDetId id(subdet, iside*etaRing, phis[k].first, depthBin);
      hcalIds.push_back(id);
      makeCell(id,hcalCells[i],phis[k].second,dphi,geom) ;
    }
  }
  
  edm::LogInfo("HCalGeom") << "Number of HCAL DetIds made for " << subdet
			   << " is " << hcalIds.size();
}

void HcalDDDGeometryLoader::makeCell(const HcalDetId& detId,
				     const HcalCellType& hcalCell,
				     double phi, double dphi,
				     CaloSubdetectorGeometry* geom) const {

  // the two eta boundaries of the cell
  double          eta1   = hcalCell.etaMin();
  double          eta2   = hcalCell.etaMax();
  HcalSubdetector subdet = detId.subdet();
  double          eta    = 0.5*(eta1+eta2) * detId.zside();
  double          deta   = (eta2-eta1);
  double          theta  = 2.0*atan(exp(-eta));

  // barrel vs forward
  bool rzType   = hcalCell.depthType();
  bool isBarrel = (subdet == HcalBarrel || subdet == HcalOuter);

  double          z, r, thickness;
#ifdef EDM_ML_DEBUG
  double          r0, r1, r2;
#endif
  if (rzType) {
    r          = hcalCell.depthMin();
    if (isBarrel) {
      z         = r * sinh(eta); // sinh(eta) == 1/tan(theta)
      thickness = (hcalCell.depthMax() - r) * cosh(eta); // cosh(eta) == 1/sin(theta)
#ifdef EDM_ML_DEBUG
      r1        = r;
      r2        = hcalCell.depthMax();
      r0        = 0.5*(r1+r2);
#endif
    } else {
      z         = r * sinh(eta2);
      thickness = 2. * hcalCell.halfSize();
      r         = z/sinh(std::abs(eta));
#ifdef EDM_ML_DEBUG
      r0        = z/sinh(std::abs(eta));
      r1        = z/sinh(std::abs(eta)+0.5*deta);
      r2        = z/sinh(std::abs(eta)-0.5*deta);
#endif
    }
#ifdef EDM_ML_DEBUG
    std::cout << "HcalDDDGeometryLoader::makeCell SubDet " << subdet
	      << " eta = " << eta << " theta = " << theta << " r = " << r 
	      << " thickness = " << thickness << " r0-r2 (" << r0 << ":" 
	      << r1 << ":" << r2 << ")" << std::endl;
#endif
  } else {
    z          = hcalCell.depthMin();
    thickness  = hcalCell.depthMax() - z;
    if (isBH_) z += (0.5*thickness);
    z         *= detId.zside(); // get the sign right.
    r          = z * tan(theta);
    thickness /= std::abs(cos(theta));
#ifdef EDM_ML_DEBUG
    r0         = z/sinh(std::abs(eta));
    r1         = z/sinh(std::abs(eta)+0.5*deta);
    r2         = z/sinh(std::abs(eta)-0.5*deta);
    std::cout << "HcalDDDGeometryLoader::makeCell SubDet " << subdet
	      << " eta = " << eta << " theta = " << theta << " z = " << z 
	      << " r = " << r << " thickness = " << thickness << " r0-r2 (" 
	      << r0 << ":" << r1 << ":" << r2 << ")" << std::endl;    
#endif
  }

  double x = r * cos(phi);
  double y = r * sin(phi);
  GlobalPoint point(x,y,z);

#ifdef EDM_ML_DEBUG
  std::cout << "HcalDDDGeometryLoader::makeCell for " << detId << " Point " 
	    << point << " deta = " << deta << " dphi = " << dphi 
	    << " thickness = " << thickness << " isBarrel = " << isBarrel 
	    << " " << rzType << std::endl;
#endif

  std::vector<CCGFloat> hp ;
  hp.reserve(3) ;
  
  if (subdet==HcalForward) {
    hp.push_back(deta/2.) ;
    hp.push_back(dphi/2.) ;
    hp.push_back(thickness/2.) ;
  } else { 
    const double sign ( isBarrel ? 1 : -1 ) ;
    hp.push_back(deta/2.) ;
    hp.push_back(dphi/2.) ;
    hp.push_back(sign*thickness/2.) ;
  }
  geom->newCell( point, point, point,
		 CaloCellGeometry::getParmPtr( hp, 
					       geom->parMgr(), 
					       geom->parVecVec() ),
		 detId ) ;
}
