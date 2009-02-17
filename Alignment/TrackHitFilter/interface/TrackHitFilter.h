// -*- C++ -*-
//
// Package:    TrackHitFilter
// Class:      TrackHitFilter
// 
/**\class TrackHitFilter TrackHitFilter.h Alignment/TrackHitFilter/interface/TrackHitFilter.h

 Description: Selects some track hits for refitting input 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Roberto Covarelli
//         Created:  Mon Jan 15 10:39:42 CET 2007
// $Id: TrackHitFilter.h,v 1.7 2008/10/13 12:42:14 ntran Exp $
//
//

// system include files
#include <memory>
#include "stdio.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/DetId/interface/DetId.h" 
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h" 

//
// class declaration
//

class TrackHitFilter : public edm::EDProducer {
   public:
      explicit TrackHitFilter(const edm::ParameterSet&);
      ~TrackHitFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
  //     bool keepThisHit(DetId id, int type, int layer);
      bool keepThisHit(DetId id, int type, int layer, const TrackingRecHit* , const edm::Event&, const edm::EventSetup&);  

   protected:
      edm::InputTag theSrc;    
      std::string theHitSel;
      unsigned int theMinHits;
      bool rejectBadMods;
      std::vector<unsigned int> theBadMods; 
  /* EM */
  bool rejectBadStoNHits;
  std::string theCMNSubtractionMode;
  double theStoNthreshold;
  /*RC*/
  bool rejectBadClusterPixelHits;
  double thePixelClusterthreshold;
};
