#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::auto_ptr<CaloSubdetectorGeometry> CaloTowerHardcodeGeometryLoader::load() {
  CaloTowerGeometry* geom=new CaloTowerGeometry();
  int n=0;
  // simple loop
  for (int ieta=-limits.lastHFRing(); ieta<=limits.lastHFRing(); ieta++) {
    if (ieta==0) continue; // skip not existing eta=0 ring
    for (int iphi=1; iphi<=72; iphi++) {
      if (abs(ieta)>=limits.firstHFQuadPhiRing() && ((iphi-1)%4)==0) continue;
      if (abs(ieta)>=limits.firstHEDoublePhiRing() && ((iphi-1)%2)!=0) continue;
      geom->addCell(CaloTowerDetId(ieta,iphi),makeCell(ieta,iphi, geom));
      n++;
    }
  }
  edm::LogInfo("Geometry") << "CaloTowersHardcodeGeometry made " << n << " towers.";
  return std::auto_ptr<CaloSubdetectorGeometry>(geom); 
}

const CaloCellGeometry* CaloTowerHardcodeGeometryLoader::makeCell(int ieta, int iphi,
								  CaloSubdetectorGeometry* geom) const {
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
  if (etaRing>limits.lastHERing()) {
    eta1 = theHFEtaBounds[etaRing-limits.firstHFRing()];
    eta2 = theHFEtaBounds[etaRing-limits.firstHFRing()+1];
  } else {
    eta1 = theHBHEEtaBounds[etaRing-1];
    eta2 = theHBHEEtaBounds[etaRing];
  }
  double eta = 0.5*(eta1+eta2);
  double deta = (eta2-eta1);  

  // in radians
  double dphi_nominal = 2.0*M_PI / limits.nPhiBins(1); // always the same
  double dphi_half = M_PI / limits.nPhiBins(etaRing); // half-width
  
  double phi_low = dphi_nominal*(iphi-1); // low-edge boundaries are constant...
  double phi = phi_low+dphi_half;

  double x,y,z,thickness;
  bool alongZ=true;
  if (etaRing>limits.lastHERing()) { // forward
    z=HFz;
    double r=z/sinh(eta);
    x=r * cos(phi);
    y=r * sin(phi);
    thickness=HFthick/tanh(eta);
  } else if (etaRing>17) { // EE-containing
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
  std::vector<double> hh ;
  hh.resize(3) ;
  hh.push_back( deta ) ;
  hh.push_back( dphi_half*2 ) ;
  hh.push_back( mysign*thickness ) ;
  class DummyClass ;
  return new calogeom::IdealObliquePrism(
     point,
     geom->cornersMgr(),
     CaloCellGeometry::getParmPtr( hh, 3, geom->parVecVec() ) );
}
