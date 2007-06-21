#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

HoECalculator::HoECalculator () :
               theCaloGeom_(0)
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

double HoECalculator::getHoE(GlobalPoint pclu, float ecalEnergy, const edm::Event& e , const edm::EventSetup& c )
{
   if ( !theCaloGeom_.isValid() )
       c.get<IdealGeometryRecord>().get(theCaloGeom_) ;

   //product the geometry
   theCaloGeom_.product() ;

   //Create a CaloRecHitMetaCollection
   edm::Handle< HBHERecHitCollection > hbhe ;
   e.getByLabel("hbhereco","",hbhe);
   const HBHERecHitCollection* hithbhe_ = hbhe.product();

   double HoE;
   const CaloGeometry& geometry = *theCaloGeom_ ;
   const CaloSubdetectorGeometry *geometry_p ; 
   geometry_p = geometry.getSubdetectorGeometry (DetId::Hcal,4) ;
   DetId hcalDetId ;
   hcalDetId = geometry_p->getClosestCell(pclu) ;
   double hcalEnergy = 0 ;
   CaloRecHitMetaCollection f;
   f.add(hithbhe_);
   CaloRecHitMetaCollection::const_iterator iterRecHit ; 
   iterRecHit = f.find(hcalDetId) ;
   hcalEnergy = iterRecHit->energy() ;
   HoE = hcalEnergy/ecalEnergy ;

   return HoE ;
}
