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
// $Id: EventWithHistoryEDFilter.cc,v 1.3 2010/01/12 09:13:04 venturia Exp $
//
//

// system include files
#include <memory>

#include <vector>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

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

class EventWithHistoryEDFilter : public edm::global::EDFilter<> {
public:
  explicit EventWithHistoryEDFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID streamId, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  std::vector<EventWithHistoryFilter> ehfilters_;
  const bool debu_;
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
EventWithHistoryEDFilter::EventWithHistoryEDFilter(const edm::ParameterSet& iConfig)
    : ehfilters_(), debu_(iConfig.getUntrackedParameter<bool>("debugPrint", false)) {
  //now do what ever initialization is needed

  std::vector<edm::ParameterSet> filterconfigs(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >(
      "filterConfigurations", std::vector<edm::ParameterSet>()));

  for (auto& ps : filterconfigs) {
    ps.augment(iConfig.getUntrackedParameter<edm::ParameterSet>("commonConfiguration", edm::ParameterSet()));

    ehfilters_.emplace_back(ps, consumesCollector());
  }
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool EventWithHistoryEDFilter::filter(edm::StreamID streamId, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  bool selected = false;

  for (const auto& filter : ehfilters_) {
    selected = selected || filter.selected(iEvent, iSetup);
  }

  if (debu_ && selected)
    edm::LogInfo("SELECTED") << "selected event";

  return selected;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventWithHistoryEDFilter);
