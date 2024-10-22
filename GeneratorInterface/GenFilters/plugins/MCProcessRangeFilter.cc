// -*- C++ -*-
//
// Package:    MCProcessRangeFilter
// Class:      MCProcessRangeFilter
//
/*

 Description: filter events based on the Pythia ProcessID and the Pt_hat

 Implementation: inherits from generic EDFilter

*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Sept 11 10:57:54 CET 2006
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <string>

class MCProcessRangeFilter : public edm::global::EDFilter<> {
public:
  explicit MCProcessRangeFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const int minProcessID;
  const int maxProcessID;
  const double pthatMin;
  const double pthatMax;
};

MCProcessRangeFilter::MCProcessRangeFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      minProcessID(iConfig.getUntrackedParameter("MinProcessID", 0)),
      maxProcessID(iConfig.getUntrackedParameter("MaxProcessID", 500)),
      pthatMin(iConfig.getUntrackedParameter("MinPthat", 0)),
      pthatMax(iConfig.getUntrackedParameter("MaxPthat", 14000)) {}

bool MCProcessRangeFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  // do the selection -- processID 0 is always accepted

  if (myGenEvent->signal_process_id() > minProcessID && myGenEvent->signal_process_id() < maxProcessID) {
    if (myGenEvent->event_scale() > pthatMin && myGenEvent->event_scale() < pthatMax) {
      accepted = true;
    }
  }
  return accepted;
}

DEFINE_FWK_MODULE(MCProcessRangeFilter);
