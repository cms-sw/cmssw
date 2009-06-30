// -*- C++ -*-
//
// Package:    PickEvents
// Class:      PickEvents
// 
/**\class PickEvents PickEvents.cc DPGAnalysis/PickEvents/src/PickEvents.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Henry Schmitt
//         Created:  Mon Sep 15 19:36:37 CEST 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class PickEvents : public edm::EDFilter {
   public:
      explicit PickEvents(const edm::ParameterSet&);
      ~PickEvents();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  int nEventsAnalyzed;
  int nEventsSelected;
  int whichRun;
  int whichEventFirst;
  int whichEventLast ;
      
};

using namespace std;
using namespace edm;



PickEvents::PickEvents(const edm::ParameterSet& iConfig)
{
  whichRun = iConfig.getUntrackedParameter<int>("whichRun",1);
  whichEventFirst = iConfig.getUntrackedParameter<int>("whichEventFirst",11);
  whichEventLast  = iConfig.getUntrackedParameter<int>("whichEventLast",19);

}


PickEvents::~PickEvents()
{
}

bool
PickEvents::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   int kRun   = iEvent.id().run();
   int kEvent = iEvent.id().event();

   bool selectThisEvent = false;
   if (kRun == whichRun) {
     if ( (kEvent >= whichEventFirst) && (kEvent <= whichEventLast) ) {
       selectThisEvent = true;
     }
   }

   nEventsAnalyzed++;
   if (selectThisEvent) nEventsSelected++;

   return selectThisEvent;
}

void 
PickEvents::beginJob(const edm::EventSetup&)
{
  nEventsAnalyzed = 0;
  nEventsSelected = 0;
}
void 
PickEvents::endJob() {
  cout << "================================================\n"
       << "  n Events Analyzed ............... " << nEventsAnalyzed << endl
       << "  n Events Selected ............... " << nEventsSelected<< endl
       << "================================================\n\n" ;
}


DEFINE_FWK_MODULE(PickEvents);
