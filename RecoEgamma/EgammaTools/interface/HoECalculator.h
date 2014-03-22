#ifndef HoECalculator_h
#define HoECalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class HoECalculator
{
  public:
  
   HoECalculator() ;
   HoECalculator(const edm::ESHandle<CaloGeometry>&) ;

   double operator() ( const reco::BasicCluster* , 
                       const edm::Event& e , 
		       const edm::EventSetup& c )  ;
   
   double operator() ( const reco::SuperCluster* , 
                       const edm::Event& e , 
		       const edm::EventSetup& c )  ;
  
   /*
   double operator() ( const reco::SuperCluster* , 
                       HBHERecHitMetaCollection *mhbhe,
		       int ialgo=1);

   double operator() ( cost reco::BasicCluster* , 
                       HBHERecHitMetaCollection *mhbhe);
   */

  private:
  
   double getHoE(GlobalPoint pos, float energy,
		 const edm::Event& e , 
		 const edm::EventSetup& c )  ;
   /*      
   double getHoE(GlobalPoint pos, float energy,
                 HBHERecHitMetaCollection *mhbhe);
   */
   
    edm::ESHandle<CaloGeometry>  theCaloGeom_ ;
    const HBHERecHitCollection* hithbhe_ ;
};

#endif
