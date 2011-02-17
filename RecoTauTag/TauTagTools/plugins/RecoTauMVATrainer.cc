/*
 * RecoTauMVATrainer
 *
 * Pass either signal or background events (with an option weight) to the
 * MVATrainer interface.
 *
 * Author: Evan K. Friis
 */

#include <boost/foreach.hpp>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "RecoTauTag/RecoTau/interface/RecoTauMVAHelper.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

class RecoTauMVATrainer : public edm::EDAnalyzer {
  public:
    explicit RecoTauMVATrainer(const edm::ParameterSet &pset);
    virtual ~RecoTauMVATrainer() {};
    virtual void analyze(const edm::Event &evt, const edm::EventSetup &es);
  private:
    reco::tau::RecoTauMVAHelper mva_;
    edm::InputTag signalSrc_;
    edm::InputTag backgroundSrc_;
    bool applyWeights_;
    edm::InputTag signalWeightsSrc_;
    edm::InputTag backgroundWeightsSrc_;
};

RecoTauMVATrainer::RecoTauMVATrainer(const edm::ParameterSet &pset)
  : mva_(pset.getParameter<std::string>("computerName"),
         pset.getParameter<std::string>("dbLabel"),
         pset.getParameter<edm::ParameterSet>("discriminantOptions")),
    signalSrc_(pset.getParameter<edm::InputTag>("signalSrc")),
    backgroundSrc_(pset.getParameter<edm::InputTag>("backgroundSrc")) {
      // Check if we want to apply weights
      applyWeights_ = false;
      if (pset.exists("signalWeights")) {
        applyWeights_ = true;
        signalWeightsSrc_ = pset.getParameter<edm::InputTag>("signalWeights");
        backgroundWeightsSrc_ = pset.getParameter<edm::InputTag>("backgroundWeights");
      }
    }

namespace {

// Upload a view to the MVA
void uploadTrainingData(reco::tau::RecoTauMVAHelper *helper,
                        const edm::Handle<reco::CandidateView>& taus,
                        const edm::Handle<reco::PFTauDiscriminator>& weights,
                        bool isSignal) {
  // Convert to a vector of refs
  reco::PFTauRefVector tauRefs =
      reco::tau::castView<reco::PFTauRefVector>(taus);
  // Loop over our taus and pass each to the MVA interface
  BOOST_FOREACH(reco::PFTauRef tau, tauRefs) {
    // Lookup the weight if desired
    double weight = (weights.isValid()) ? (*weights)[tau] : 1.0;
    helper->train(tau, isSignal, weight);
  }
}

}


void RecoTauMVATrainer::analyze(const edm::Event &evt,
                                const edm::EventSetup &es) {
  // Make sure the MVA is up to date from the DB
  mva_.setEvent(evt, es);

  // Get a view to our taus
  edm::Handle<reco::CandidateView> signal;
  edm::Handle<reco::CandidateView> background;

  bool signalExists = true;
  try {
    evt.getByLabel(signalSrc_, signal);
    if (!signal.isValid())
      signalExists = false;
  } catch(...) {
    signalExists = false;
  }

  bool backgroundExists = true;
  try {
    evt.getByLabel(backgroundSrc_, background);
    if (!background.isValid())
      backgroundExists = false;
  } catch(...) {
    backgroundExists = false;
  }

  // Get weights if desired
  edm::Handle<reco::PFTauDiscriminator> signalWeights;
  edm::Handle<reco::PFTauDiscriminator> backgroundWeights;
  if (applyWeights_ && signalExists)
    evt.getByLabel(signalWeightsSrc_, signalWeights);
  if (applyWeights_ && backgroundExists)
    evt.getByLabel(backgroundWeightsSrc_, backgroundWeights);

  if (signalExists)
    uploadTrainingData(&mva_, signal, signalWeights, true);
  if (backgroundExists)
    uploadTrainingData(&mva_, background, backgroundWeights, false);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauMVATrainer);
