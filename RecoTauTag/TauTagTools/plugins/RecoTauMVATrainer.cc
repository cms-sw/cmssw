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

#include "CommonTools/Utils/interface/StringObjectFunction.h"

class RecoTauMVATrainer : public edm::EDAnalyzer {
  public:
    explicit RecoTauMVATrainer(const edm::ParameterSet &pset);
    virtual ~RecoTauMVATrainer() {};
    virtual void analyze(const edm::Event &evt, const edm::EventSetup &es);
  private:
    typedef std::vector<edm::InputTag> VInputTag;
    reco::tau::RecoTauMVAHelper mva_;
    edm::InputTag signalSrc_;
    edm::InputTag backgroundSrc_;
    bool applyWeights_;
    edm::InputTag signalWeightsSrc_;
    edm::InputTag backgroundWeightsSrc_;
    VInputTag eventWeights_;

    std::string signalWeightFunction_;
    std::string backgroundWeightFunction_;

    std::auto_ptr<StringObjectFunction<reco::PFTau> >
      signalWeightFunctionImpl_;

    std::auto_ptr<StringObjectFunction<reco::PFTau> >
      backgroundWeightFunctionImpl_;
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
  if (pset.exists("eventWeights")) {
    eventWeights_ = pset.getParameter<VInputTag>("eventWeights");
  }

  // Check if we want to apply a string based weight
  backgroundWeightFunction_ = pset.exists("backgroundWeightFunction") ?
    pset.getParameter<std::string>("backgroundWeightFunction") : "null";
  signalWeightFunction_ = pset.exists("signalWeightFunction") ?
    pset.getParameter<std::string>("signalWeightFunction") : "null";

  if (backgroundWeightFunction_ != "null") {
    backgroundWeightFunctionImpl_.reset(
        new StringObjectFunction<reco::PFTau>(backgroundWeightFunction_));
  }

  if (signalWeightFunction_ != "null") {
    signalWeightFunctionImpl_.reset(
        new StringObjectFunction<reco::PFTau>(signalWeightFunction_));
  }
}

namespace {

// Upload a view to the MVA
void uploadTrainingData(reco::tau::RecoTauMVAHelper *helper,
                        const edm::Handle<reco::CandidateView>& taus,
                        const edm::Handle<reco::PFTauDiscriminator>& weights,
                        const StringObjectFunction<reco::PFTau>* stringWeight,
                        bool isSignal, double eventWeight) {
  // Convert to a vector of refs
  reco::PFTauRefVector tauRefs =
      reco::tau::castView<reco::PFTauRefVector>(taus);
  // Loop over our taus and pass each to the MVA interface
  BOOST_FOREACH(reco::PFTauRef tau, tauRefs) {
    // Lookup the weight if desired
    double totalWeight = eventWeight;

    if (stringWeight) {
      totalWeight *= (*stringWeight)(*tau);
    }

    if (weights.isValid()) {
      totalWeight *= (*weights)[tau];
    }

    helper->train(tau, isSignal, totalWeight);
  }
}

}


void RecoTauMVATrainer::analyze(const edm::Event &evt,
                                const edm::EventSetup &es) {
  // Get a view to our taus
  edm::Handle<reco::CandidateView> signal;
  edm::Handle<reco::CandidateView> background;

  bool signalExists = false;
  evt.getByLabel(signalSrc_, signal);
  if (signal.isValid() && signal->size())
    signalExists = true;

  bool backgroundExists = false;
  evt.getByLabel(backgroundSrc_, background);
  if (background.isValid() && background->size())
    backgroundExists = true;

  // Check if we have anything to do
  bool somethingToDo = signalExists || backgroundExists;
  if (!somethingToDo)
    return;

  // Make sure the MVA is up to date from the DB
  mva_.setEvent(evt, es);

  // Get event weights if specified
  double eventWeight = 1.0;
  BOOST_FOREACH(const edm::InputTag& weightTag, eventWeights_) {
    edm::Handle<double> weightHandle;
    evt.getByLabel(weightTag, weightHandle);
    if (weightHandle.isValid())
      eventWeight *= *weightHandle;
  }

  // Get weights if desired
  edm::Handle<reco::PFTauDiscriminator> signalWeights;
  edm::Handle<reco::PFTauDiscriminator> backgroundWeights;
  if (applyWeights_ && signalExists)
    evt.getByLabel(signalWeightsSrc_, signalWeights);
  if (applyWeights_ && backgroundExists)
    evt.getByLabel(backgroundWeightsSrc_, backgroundWeights);

  if (signalExists)
    uploadTrainingData(&mva_, signal, signalWeights,
        signalWeightFunctionImpl_.get(), true, eventWeight);
  if (backgroundExists)
    uploadTrainingData(&mva_, background, backgroundWeights,
        backgroundWeightFunctionImpl_.get(), false, eventWeight);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauMVATrainer);
