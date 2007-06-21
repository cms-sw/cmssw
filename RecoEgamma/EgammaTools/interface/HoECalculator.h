#ifndef HoECalculator_h
#define HoECalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollection.h"

class HoECalculator
{
  public:
  
   HoECalculator() ;

   double operator() ( const reco::BasicCluster* , 
                       const edm::Event& e , 
		       const edm::EventSetup& c )  ;
   
   double operator() ( const reco::SuperCluster* , 
                       const edm::Event& e , 
		       const edm::EventSetup& c )  ;
   
  private:
  
   double getHoE(GlobalPoint pos, float energy,
		 const edm::Event& e , 
		 const edm::EventSetup& c )  ;
   
    edm::ESHandle<CaloGeometry>  theCaloGeom_ ;
    const HBHERecHitCollection* hithbhe_ ;
};

#endif
