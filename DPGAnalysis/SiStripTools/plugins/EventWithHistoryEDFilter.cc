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
// $Id: EventWithHistoryEDFilter.cc,v 1.4 2013/02/27 19:49:46 wmtan Exp $
//
//


// system include files
#include <memory>

#include <vector>
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
  
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  std::vector<EventWithHistoryFilter> _ehfilters;
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
  _ehfilters(),
  _debu(iConfig.getUntrackedParameter<bool>("debugPrint",false))
{
   //now do what ever initialization is needed

  std::vector<edm::ParameterSet> filterconfigs(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >
					       ("filterConfigurations",std::vector<edm::ParameterSet>()));
  
  for(std::vector<edm::ParameterSet>::iterator ps=filterconfigs.begin();
      ps!=filterconfigs.end();++ps) {

    ps->augment(iConfig.getUntrackedParameter<edm::ParameterSet>("commonConfiguration",edm::ParameterSet()));
    
    const EventWithHistoryFilter filter(*ps);
    _ehfilters.push_back(filter);

  }


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

  bool selected = false;

  for(std::vector<EventWithHistoryFilter>::const_iterator filter=_ehfilters.begin();
      filter!=_ehfilters.end();++filter) {

    selected = selected || filter->selected(iEvent,iSetup);

  }

  if(_debu && selected ) edm::LogInfo("SELECTED") << "selected event";
 
  return selected;

}

// ------------ method called once each job just before starting event loop  ------------
void 
EventWithHistoryEDFilter::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void 
EventWithHistoryEDFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventWithHistoryEDFilter);
