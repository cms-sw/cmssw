#ifndef RxCalculator_h
#define RxCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"


#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class RxCalculator
{
  public:
  
   RxCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::InputTag& hbheLabel,const edm::InputTag& hfLabel,const edm::InputTag& hoLabel) ;
  
   double getRx (const reco::SuperClusterRef clus, double i, double threshold, double innerR=0.0);
   double getRFx(const reco::SuperClusterRef clus, double i, double threshold);
   double getROx(const reco::SuperClusterRef clus, double i, double threshold);
   double getCRx(const reco::SuperClusterRef clus, double i, double threshold, double innerR=0.0); // background subtracted Rx

  private:
   
   const HBHERecHitCollection         *fHBHERecHits_;
   const HORecHitCollection           *fHORecHits_;
   const HFRecHitCollection           *fHFRecHits_;
   const CaloGeometry                 *geometry_;
};

#endif
