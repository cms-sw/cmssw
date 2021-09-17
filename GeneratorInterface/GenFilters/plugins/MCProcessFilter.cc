// -*- C++ -*-
//
// Package:    MCProcessFilter
// Class:      MCProcessFilter
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

#include <iostream>
#include <string>
#include <vector>

class MCProcessFilter : public edm::global::EDFilter<> {
public:
  explicit MCProcessFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> processID;
  std::vector<double> pthatMin;
  std::vector<double> pthatMax;
};

using namespace std;

MCProcessFilter::MCProcessFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))) {
  vector<int> defproc;
  defproc.push_back(0);
  processID = iConfig.getUntrackedParameter<vector<int> >("ProcessID", defproc);
  vector<double> defpthatmin;
  defpthatmin.push_back(0.);
  pthatMin = iConfig.getUntrackedParameter<vector<double> >("MinPthat", defpthatmin);
  vector<double> defpthatmax;
  defpthatmax.push_back(10000.);
  pthatMax = iConfig.getUntrackedParameter<vector<double> >("MaxPthat", defpthatmax);

  // checkin size of phthat vectors -- default is allowed
  if ((pthatMin.size() > 1 && processID.size() != pthatMin.size()) ||
      (pthatMax.size() > 1 && processID.size() != pthatMax.size())) {
    cout << "WARNING: MCPROCESSFILTER : size of MinPthat and/or MaxPthat not matching with ProcessID size!!" << endl;
  }

  // if pthatMin size smaller than processID , fill up further with defaults
  if (processID.size() > pthatMin.size()) {
    vector<double> defpthatmin2;
    for (unsigned int i = 0; i < processID.size(); i++) {
      defpthatmin2.push_back(0.);
    }
    pthatMin = defpthatmin2;
  }
  // if pthatMax size smaller than processID , fill up further with defaults
  if (processID.size() > pthatMax.size()) {
    vector<double> defpthatmax2;
    for (unsigned int i = 0; i < processID.size(); i++) {
      defpthatmax2.push_back(10000.);
    }
    pthatMax = defpthatmax2;
  }
}

bool MCProcessFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  // do the selection -- processID 0 is always accepted
  for (unsigned int i = 0; i < processID.size(); i++) {
    if (processID[i] == myGenEvent->signal_process_id() || processID[i] == 0) {
      if (myGenEvent->event_scale() > pthatMin[i] && myGenEvent->event_scale() < pthatMax[i]) {
        accepted = true;
      }
    }
  }
  return accepted;
}

DEFINE_FWK_MODULE(MCProcessFilter);
