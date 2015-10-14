#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

typedef CaloCellGeometry::CCGFloat CCGFloat ;

std::auto_ptr<CaloSubdetectorGeometry> CaloTowerHardcodeGeometryLoader::load(const HcalTopology *hcaltopo, const HcalDDDRecConstants* hcons) {

  m_hcaltopo = hcaltopo;
  m_hcons = hcons;

  //get eta limits from hcal rec constants
  theHFEtaBounds   = m_hcons->getEtaTableHF();
  theHBHEEtaBounds = m_hcons->getEtaTable();

  CaloTowerGeometry* geom=new CaloTowerGeometry();
  int nnn(0);
  // simple loop
  for (int ieta=-m_hcaltopo->lastHFRing(); ieta<=m_hcaltopo->lastHFRing(); ieta++) {
    if (ieta==0) continue; // skip not existing eta=0 ring
    for (int iphi=1; iphi<=72; iphi++) {
      if (abs(ieta)>=m_hcaltopo->firstHFQuadPhiRing() && ((iphi-1)%4)==0) continue;
      if (abs(ieta)>=m_hcaltopo->firstHEDoublePhiRing() && ((iphi-1)%2)!=0) continue;
      ++nnn;
    }
  }
  std::cout << "Number of corners" << nnn << std::endl;
  if( 0 == geom->cornersMgr() ) geom->allocateCorners (nnn); 
  if( 0 == geom->parMgr() ) geom->allocatePar (
     CaloTowerGeometry::k_NumberOfParametersPerShape*CaloTowerGeometry::k_NumberOfShapes,
     CaloTowerGeometry::k_NumberOfParametersPerShape ) ;

  int n=0;
  // simple loop
  for (int ieta=-m_hcaltopo->lastHFRing(); ieta<=m_hcaltopo->lastHFRing(); ieta++) {
    if (ieta==0) continue; // skip not existing eta=0 ring
    for (int iphi=1; iphi<=72; iphi++) {
      if (abs(ieta)>=m_hcaltopo->firstHFQuadPhiRing() && ((iphi-1)%4)==0) continue;
      if (abs(ieta)>=m_hcaltopo->firstHEDoublePhiRing() && ((iphi-1)%2)!=0) continue;
      makeCell(ieta,iphi, geom);
      n++;
    }
  }
  edm::LogInfo("Geometry") << "CaloTowersHardcodeGeometry made " << n << " towers.";
  std::cout << "End from CaloTowerGeometryLoader\n";
  return std::auto_ptr<CaloSubdetectorGeometry>(geom); 
}

void
CaloTowerHardcodeGeometryLoader::makeCell( int ieta,
					   int iphi,
					   CaloSubdetectorGeometry* geom ) const {
  const double EBradius = 143.0; // cm
  const double HOradius = 406.0+1.0;
  const double EEz = 320.0; // rough (cm)
  const double HEz = 568.0; // back (cm)
  const double HFz = 1100.0;
  const double HFthick = 165;
  // Tower 17 is the last EB tower

  int etaRing=abs(ieta);
  int sign=(ieta>0)?(1):(-1);
  double eta1, eta2;
  if (abs(ieta)>m_hcaltopo->lastHERing()) {
    eta1 = theHFEtaBounds[etaRing-m_hcaltopo->firstHFRing()];
    eta2 = theHFEtaBounds[etaRing-m_hcaltopo->firstHFRing()+1];
  } else {
    eta1 = theHBHEEtaBounds[etaRing-1];
    eta2 = theHBHEEtaBounds[etaRing];
  }
  double eta = 0.5*(eta1+eta2);
  double deta = (eta2-eta1);  

  // in radians
  double dphi_nominal = 2.0*M_PI / m_hcaltopo->nPhiBins(1); // always the same
  double dphi_half = M_PI / m_hcaltopo->nPhiBins(etaRing); // half-width
  
  double phi_low = dphi_nominal*(iphi-1); // low-edge boundaries are constant...
  double phi = phi_low+dphi_half;

  double x,y,z,thickness;
  bool alongZ=true;
  if (abs(ieta)>m_hcaltopo->lastHERing()) { // forward
    z=HFz;
    double r=z/sinh(eta);
    x=r * cos(phi);
    y=r * sin(phi);
    thickness=HFthick/tanh(eta);
  } else if (abs(ieta)>m_hcaltopo->firstHERing()+1) { // EE-containing
    z=EEz;
    double r=z/sinh(eta);
    x=r * cos(phi);
    y=r * sin(phi);
    thickness=(HEz-EEz)/tanh(eta);
  } else { // EB-containing
    x=EBradius * cos(phi);
    y=EBradius * sin(phi);
    alongZ=false;
    z=EBradius * sinh(eta);
    thickness=(HOradius-EBradius) * cosh(eta);
  }

  z*=sign;
  GlobalPoint point(x,y,z);

  const double mysign ( !alongZ ? 1 : -1 ) ;
  std::vector<CCGFloat> hh ;
  hh.reserve(5) ;
  hh.push_back( deta/2 ) ;
  hh.push_back( dphi_half ) ;
  hh.push_back( mysign*thickness/2. ) ;

  hh.push_back( fabs( eta ) ) ;
  hh.push_back( fabs( z ) ) ;

  geom->newCell( point, point, point,
		 CaloCellGeometry::getParmPtr( hh, 
					       geom->parMgr(), 
					       geom->parVecVec() ),
		 CaloTowerDetId( ieta, iphi ) ) ;
}
