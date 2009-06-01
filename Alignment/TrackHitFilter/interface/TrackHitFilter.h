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
// $Id: TrackHitFilter.h,v 1.5 2007/12/06 01:57:45 ratnik Exp $
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
      bool keepThisHit(DetId id, int type, int layer);

   protected:
      edm::InputTag theSrc;    
      std::string theHitSel;
      unsigned int theMinHits;
      bool rejectBadMods;
      std::vector<unsigned int> theBadMods; 
};
