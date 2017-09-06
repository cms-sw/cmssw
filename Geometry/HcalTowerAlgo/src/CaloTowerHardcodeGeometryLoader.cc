#include "Geometry/HcalTowerAlgo/interface/CaloTowerHardcodeGeometryLoader.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iterator>
#include <algorithm>

//#define DebugLog

typedef CaloCellGeometry::CCGFloat CCGFloat ;

std::unique_ptr<CaloSubdetectorGeometry> CaloTowerHardcodeGeometryLoader::load(const CaloTowerTopology *limits, const HcalTopology *hcaltopo, const HcalDDDRecConstants* hcons) {

  m_limits = limits;
  m_hcaltopo = hcaltopo;
  m_hcons = hcons;

  //get eta limits from hcal rec constants
  theHFEtaBounds   = m_hcons->getEtaTableHF();
  theHBHEEtaBounds = m_hcons->getEtaTable();

#ifdef DebugLog
  std::cout << "CaloTowerHardcodeGeometryLoader: theHBHEEtaBounds = ";
  std::copy(theHBHEEtaBounds.begin(), theHBHEEtaBounds.end(), std::ostream_iterator<double>(std::cout, ","));
  std::cout << std::endl;

  std::cout << "CaloTowerHardcodeGeometryLoader: lastHBRing = " << m_limits->lastHBRing() << ", lastHERing = " << m_limits->lastHERing() << std::endl;
  std::cout << "CaloTowerHardcodeGeometryLoader: HcalTopology: firstHBRing = " << hcaltopo->firstHBRing() << ", lastHBRing = " << hcaltopo->lastHBRing() << ", firstHERing = " << hcaltopo->firstHERing() << ", lastHERing = " << hcaltopo->lastHERing() << ", firstHFRing = " << hcaltopo->firstHFRing() << ", lastHFRing = " << hcaltopo->lastHFRing() << std::endl;
#endif

  CaloTowerGeometry* geom=new CaloTowerGeometry(m_limits);

  if( nullptr == geom->cornersMgr() ) geom->allocateCorners ( 
     geom->numberOfCellsForCorners() ) ;
  if( nullptr == geom->parMgr() ) geom->allocatePar (
     geom->numberOfParametersPerShape()*geom->numberOfShapes(),
     geom->numberOfParametersPerShape() ) ;

  // simple loop
  for (uint32_t din = 0; din < m_limits->sizeForDenseIndexing(); ++din) {
    makeCell(din, geom);
  }
  edm::LogInfo("Geometry") << "CaloTowersHardcodeGeometry made " << m_limits->sizeForDenseIndexing() << " towers.";
  return std::unique_ptr<CaloSubdetectorGeometry>(geom); 
}

void
CaloTowerHardcodeGeometryLoader::makeCell( uint32_t din,
					   CaloSubdetectorGeometry* geom ) const {
  const double EBradius = 143.0; // cm
  const double HOradius = 406.0+1.0;
  const double EEz = 320.0; // rough (cm)
  const double HEz = 568.0; // back (cm)
  const double HFz = 1100.0;
  const double HFthick = 165;
  // Tower 17 is the last EB tower

  //use CT topology to get the DetId for this dense index
  CaloTowerDetId id = m_limits->detIdFromDenseIndex(din);
  int ieta = id.ieta();
  int iphi = id.iphi();
  
  //use CT topology to get proper ieta for hcal
  int etaRing=m_limits->convertCTtoHcal(abs(ieta));
  int sign=(ieta>0)?(1):(-1);
  double eta1, eta2;

#ifdef DebugLog
  std::cout << "CaloTowerHardcodeGeometryLoader: ieta = " << ieta << ", iphi = " << iphi << ", etaRing = " << etaRing << std::endl;
#endif

  if (abs(ieta)>m_limits->lastHERing()) {
    eta1 = theHFEtaBounds.at(etaRing-m_hcaltopo->firstHFRing());
    eta2 = theHFEtaBounds.at(etaRing-m_hcaltopo->firstHFRing()+1);
  } else {
    eta1 = theHBHEEtaBounds.at(etaRing-1);
    eta2 = theHBHEEtaBounds.at(etaRing);
  }
  double eta = 0.5*(eta1+eta2);
  double deta = (eta2-eta1);  

  // in radians
  double dphi_nominal = 2.0*M_PI / m_hcaltopo->nPhiBins(1); // always the same
  double dphi_half = M_PI / m_hcaltopo->nPhiBins(etaRing); // half-width
  
  double phi_low = dphi_nominal*(iphi-1); // low-edge boundaries are constant...
  double phi = phi_low+dphi_half;

#ifdef DebugLog
  std::cout << "CaloTowerHardcodeGeometryLoader: eta1 = " << eta1 << ", eta2 = " << eta2 <<", eta = " << eta << ", phi = " << phi << std::endl;
#endif

  double x,y,z,thickness;
  bool alongZ=true;
  if (abs(ieta)>m_limits->lastHERing()) { // forward
    z=HFz;
    double r=z/sinh(eta);
    x=r * cos(phi);
    y=r * sin(phi);
    thickness=HFthick/tanh(eta);
  } else if (abs(ieta)>m_limits->firstHERing()+1) { // EE-containing
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
  hh.emplace_back( deta/2 ) ;
  hh.emplace_back( dphi_half ) ;
  hh.emplace_back( mysign*thickness/2. ) ;

  hh.emplace_back( fabs( eta ) ) ;
  hh.emplace_back( fabs( z ) ) ;

#ifdef DebugLog
  std::cout << "CaloTowerHardcodeGeometryLoader: x = " << x << ", y = " << y << ", z = " << z << ", thickness = " << thickness << std::endl;
#endif

  geom->newCell( point, point, point,
		 CaloCellGeometry::getParmPtr( hh, 
					       geom->parMgr(), 
					       geom->parVecVec() ),
		 id ) ;
}
