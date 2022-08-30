// -*- C++ -*-
//
// Package:    Calibration/HcalAlCaRecoProducers/AlCaGammaJetSelector
// Class:      AlCaGammaJetSelector
//
/**\class AlCaGammaJetSelector AlCaGammaJetSelector.cc Calibration/HcalAlCaRecoProducers/AlCaGammaJetSelector/src/AlCaGammaJetSelector.cc

 Description: Enable filtering of good events based on the AlCaGammaJetProducer info

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrius Juodagalvis
//         Created:  Fri, 15 Aug 2015 12:09:55 GMT
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
//#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

//
// class declaration
//

namespace alCaGammaJetSelector {
  struct Counters {
    Counters() : nProcessed_(0), nSelected_(0) {}
    mutable std::atomic<unsigned int> nProcessed_, nSelected_;
  };
}  // namespace alCaGammaJetSelector

class AlCaGammaJetSelector : public edm::stream::EDFilter<edm::GlobalCache<alCaGammaJetSelector::Counters> > {
public:
  explicit AlCaGammaJetSelector(const edm::ParameterSet&, const alCaGammaJetSelector::Counters* count);
  ~AlCaGammaJetSelector() override = default;

  static std::unique_ptr<alCaGammaJetSelector::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<alCaGammaJetSelector::Counters>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  static void globalEndJob(const alCaGammaJetSelector::Counters* counters);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}
  bool select(const reco::PhotonCollection&, const reco::PFJetCollection&);

  // ----------member data ---------------------------

  unsigned int nProcessed_, nSelected_;

  edm::InputTag labelPhoton_, labelPFJet_;
  double minPtJet_, minPtPhoton_;
  edm::EDGetTokenT<reco::PhotonCollection> tok_Photon_;
  edm::EDGetTokenT<reco::PFJetCollection> tok_PFJet_;
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
AlCaGammaJetSelector::AlCaGammaJetSelector(const edm::ParameterSet& iConfig,
                                           const alCaGammaJetSelector::Counters* counters) {
  nProcessed_ = 0;
  nSelected_ = 0;

  // get input
  labelPhoton_ = iConfig.getParameter<edm::InputTag>("PhoInput");
  labelPFJet_ = iConfig.getParameter<edm::InputTag>("PFjetInput");
  minPtJet_ = iConfig.getParameter<double>("MinPtJet");
  minPtPhoton_ = iConfig.getParameter<double>("MinPtPhoton");

  // Register consumption
  tok_Photon_ = consumes<reco::PhotonCollection>(labelPhoton_);
  tok_PFJet_ = consumes<reco::PFJetCollection>(labelPFJet_);
}

//
// member functions
//

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaGammaJetSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PhoInput", edm::InputTag("gedPhotons"));
  desc.add<edm::InputTag>("PFjetInput", edm::InputTag("ak4PFJetsCHS"));
  desc.add<double>("MinPtJet", 10.0);
  desc.add<double>("MinPtPhoton", 10.0);
  descriptions.add("alcaGammaJetSelector", desc);
}

// ------------ method called on each new Event  ------------
bool AlCaGammaJetSelector::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nProcessed_++;

  // Access the collections from iEvent
  auto const& phoHandle = iEvent.getHandle(tok_Photon_);
  if (!phoHandle.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get the product " << labelPhoton_;
    return false;  // do not filter
  }
  const reco::PhotonCollection photons = *(phoHandle.product());

  auto const& pfjetHandle = iEvent.getHandle(tok_PFJet_);
  if (!pfjetHandle.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelPFJet_;
    return false;  // do not filter
  }
  const reco::PFJetCollection pfjets = *(pfjetHandle.product());

  // Check the conditions for a good event
  if (!(select(photons, pfjets)))
    return false;

  edm::LogVerbatim("AlCaGammaJet") << "good event\n";
  nSelected_++;
  return true;
}

void AlCaGammaJetSelector::endStream() {
  globalCache()->nProcessed_ += nProcessed_;
  globalCache()->nSelected_ += nSelected_;
}

// ------------ method called once each job just after ending the event loop  ------------

void AlCaGammaJetSelector::globalEndJob(const alCaGammaJetSelector::Counters* count) {
  edm::LogWarning("AlCaGammaJet") << "Finds " << count->nSelected_ << " good events out of " << count->nProcessed_;
}

bool AlCaGammaJetSelector::select(const reco::PhotonCollection& photons, const reco::PFJetCollection& jets) {
  // Check the requirement for minimum pT
  if (photons.empty())
    return false;
  bool ok(false);
  for (reco::PFJetCollection::const_iterator itr = jets.begin(); itr != jets.end(); ++itr) {
    if (itr->pt() >= minPtJet_) {
      ok = true;
      break;
    }
  }
  if (!ok)
    return ok;
  for (reco::PhotonCollection::const_iterator itr = photons.begin(); itr != photons.end(); ++itr) {
    if (itr->pt() >= minPtPhoton_)
      return ok;
  }
  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlCaGammaJetSelector);
