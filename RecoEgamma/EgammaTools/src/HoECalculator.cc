#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

HoECalculator::HoECalculator () :
               theCaloGeom_(0)
{
} 
HoECalculator::HoECalculator (const edm::ESHandle<CaloGeometry>  &caloGeom) :
               theCaloGeom_(caloGeom)
{
} 

double HoECalculator::operator() ( const reco::BasicCluster* clus , const edm::Event& e , const edm::EventSetup& c) 
{
  return getHoE(GlobalPoint(clus->x(),clus->y(),clus->z()), clus->energy(), e,c);
}

double HoECalculator::operator() ( const reco::SuperCluster* clus , const edm::Event& e , const edm::EventSetup& c) 
{
  return getHoE(GlobalPoint(clus->x(),clus->y(),clus->z()), clus->energy(), e,c);
}

/*
double HoECalculator::operator() ( const reco::SuperCluster* clus, 
				   HBHERecHitMetaCollection *mhbhe, int ialgo) {
  double HoE=0.;
  switch (ialgo) {
    case 1:
      for (reco::CaloCluster_iterator bc=clus->clustersBegin(); bc!=clus->clustersEnd(); bc++) {
	double HoEi = getHoE(GlobalPoint((*bc)->x(),(*bc)->y(),(*bc)->z()),clus->energy(), mhbhe);
	if (HoEi > HoE) HoE = HoEi;
      }
      break;
    case 2:
      HoE = getHoE(GlobalPoint(clus->x(),clus->y(),clus->z()), clus->energy(), mhbhe);
      break;
    default:
      std::cout << "!!! algo for HoE should be 1 or 2 " << std::endl;
  } 
  return HoE;   
}

double HoECalculator::operator() ( const reco::BasicCluster* clus, 
			     HBHERecHitMetaCollection *mhbhe) {
  return getHoE(GlobalPoint(clus->x(),clus->y(),clus->z()), clus->energy(), mhbhe);
}

*/

double HoECalculator::getHoE(GlobalPoint pclu, float ecalEnergy, const edm::Event& e , const edm::EventSetup& c )
{
  if ( !theCaloGeom_.isValid() )
    c.get<CaloGeometryRecord>().get(theCaloGeom_) ;

  //product the geometry
  theCaloGeom_.product() ;

  //Create a HBHERecHitCollection
  edm::Handle< HBHERecHitCollection > hbhe ;
  e.getByLabel("hbhereco","",hbhe);
  const HBHERecHitCollection* hithbhe_ = hbhe.product();

  double HoE=0.;
  const CaloGeometry& geometry = *theCaloGeom_ ;
  const CaloSubdetectorGeometry *geometry_p ; 
  geometry_p = geometry.getSubdetectorGeometry (DetId::Hcal,4) ;
  DetId hcalDetId ;
  hcalDetId = geometry_p->getClosestCell(pclu) ;
  double hcalEnergy = 0 ;

  HBHERecHitCollection::const_iterator iterRecHit ; 
  iterRecHit = hithbhe_->find(hcalDetId) ;
  if (iterRecHit!=hithbhe_->end()) {
    hcalEnergy = iterRecHit->energy() ;
    HoE = hcalEnergy/ecalEnergy ;
  }

  return HoE ;
}

/*
double HoECalculator::getHoE(GlobalPoint pos, float energy,
			     HBHERecHitMetaCollection *mhbhe) {
  
  double HoE=0.;
  
  if (mhbhe) {
    const CaloSubdetectorGeometry *geometry_p ; 
    geometry_p =  theCaloGeom_->getSubdetectorGeometry (DetId::Hcal,4) ;
    HcalDetId dB= geometry_p->getClosestCell(pos);
    CaloRecHitMetaCollectionV::const_iterator i=mhbhe->find(dB);
    if (i!=mhbhe->end()) {
      HoE =  i->energy()/energy;
    }
  }
  return HoE ;
}
*/
