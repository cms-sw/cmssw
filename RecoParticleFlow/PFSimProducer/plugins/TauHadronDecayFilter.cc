// -*- C++ -*-
//
// Package:    TauHadronDecayFilter
// Class:      TauHadronDecayFilter
//
/**\class TauHadronDecayFilter 

 Description: filters single tau events with a tau decaying hadronically
*/
//
// Original Author:  Colin BERNET
//         Created:  Mon Nov 13 11:06:39 CET 2006
//
//

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <iostream>
#include <memory>

class TauHadronDecayFilter : public edm::EDFilter {
public:
  explicit TauHadronDecayFilter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::ParameterSet particleFilter_;
  std::unique_ptr<FSimEvent> mySimEvent;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauHadronDecayFilter);

void TauHadronDecayFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // tauHadronDecayFilter
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("particles", edm::InputTag("particleFlowBlock"));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<double>("etaMax", 10.0);
    psd0.add<double>("pTMin", 0.0);
    psd0.add<double>("EMin", 0.0);
    desc.add<edm::ParameterSetDescription>("ParticleFilter", psd0);
  }
  descriptions.add("tauHadronDecayFilter", desc);
}

using namespace edm;
using namespace std;

TauHadronDecayFilter::TauHadronDecayFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed

  particleFilter_ = iConfig.getParameter<edm::ParameterSet>("ParticleFilter");

  mySimEvent = std::make_unique<FSimEvent>(particleFilter_);
}

bool TauHadronDecayFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  Handle<vector<SimTrack> > simTracks;
  iEvent.getByLabel("g4SimHits", simTracks);
  Handle<vector<SimVertex> > simVertices;
  iEvent.getByLabel("g4SimHits", simVertices);

  mySimEvent->fill(*simTracks, *simVertices);

  if (mySimEvent->nTracks() >= 2) {
    FSimTrack& gene = mySimEvent->track(0);
    if (abs(gene.type()) != 15) {
      // first particle is not a tau.
      // -> do not filter
      return true;
    }

    FSimTrack& decayproduct = mySimEvent->track(1);
    switch (abs(decayproduct.type())) {
      case 11:  // electrons
      case 13:  // muons
        LogWarning("PFProducer") << "TauHadronDecayFilter: selecting single tau events with hadronic decay." << endl;
        // mySimEvent->print();
        return false;
      default:
        return true;
    }
  }

  // more than 2 particles
  return true;
}

void TauHadronDecayFilter::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  // init Particle data table (from Pythia)
  edm::ESHandle<HepPDT::ParticleDataTable> pdt;
  // edm::ESHandle < DefaultConfig::ParticleDataTable > pdt;

  es.getData(pdt);
  mySimEvent->initializePdt(&(*pdt));
}
