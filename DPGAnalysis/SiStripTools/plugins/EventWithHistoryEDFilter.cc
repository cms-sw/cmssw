// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      EventWithHistoryEDFilter
// 
/**\class EventWithHistoryEDFilter EventWithHistoryEDFilter.cc DPGAnalysis/SiStripTools/plugins/EventWithHistoryEDFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Dec  9 18:33:42 CET 2008
// $Id: EventWithHistoryEDFilter.cc,v 1.2 2009/03/23 10:30:38 venturia Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistoryFilter.h"

//
// class declaration
//

class EventWithHistoryEDFilter : public edm::EDFilter {
public:
  explicit EventWithHistoryEDFilter(const edm::ParameterSet&);
  ~EventWithHistoryEDFilter();
  
private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  EventWithHistoryFilter _ehfilter;
  bool _debu;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EventWithHistoryEDFilter::EventWithHistoryEDFilter(const edm::ParameterSet& iConfig):
  _ehfilter(iConfig),
  _debu(iConfig.getUntrackedParameter<bool>("debugPrint",false))
{
   //now do what ever initialization is needed

}


EventWithHistoryEDFilter::~EventWithHistoryEDFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
EventWithHistoryEDFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool selected = _ehfilter.selected(iEvent,iSetup);
  if(_debu && selected ) edm::LogInfo("SELECTED") << "selected event";
 
  return selected;

}

// ------------ method called once each job just before starting event loop  ------------
void 
EventWithHistoryEDFilter::beginJob(const edm::EventSetup&)
{}

// ------------ method called once each job just after ending the event loop  ------------
void 
EventWithHistoryEDFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventWithHistoryEDFilter);
