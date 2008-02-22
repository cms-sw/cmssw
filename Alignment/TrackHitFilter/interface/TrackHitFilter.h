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
// $Id: TrackHitFilter.h,v 1.3 2007/06/22 08:20:36 covarell Exp $
//
//

// system include files
#include <memory>
#include "stdio.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

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
      bool keepThisHit(int type, int layer);

   protected:
      edm::InputTag theSrc;    
      std::string theHitSel;
      unsigned int theMinHits;

};
