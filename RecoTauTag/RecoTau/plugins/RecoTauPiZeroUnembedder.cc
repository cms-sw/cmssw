/*
 * "Unembed" the pizeros in a reco::PFTau.
 *
 * This converts a collection of PFTaus which have their PiZeros stored
 * as std::vector<RecoTauPiZeros>s to an output collection which has
 * the PiZeros stored in a separate product, with the PiZeros stored as Refs
 * within the tau.  This will improve the de-serialization speed of the taus.
 *
 * Author: Evan K. Friis, UW Madison
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/TauReco/interface/RecoTauPiZeroFwd.h"

class RecoTauPiZeroUnembedder : public edm::stream::EDProducer<> {
  public:
    RecoTauPiZeroUnembedder(const edm::ParameterSet& pset);
    virtual ~RecoTauPiZeroUnembedder(){}
    void produce(edm::Event& evt, const edm::EventSetup& es) override;
  private:
    edm::InputTag src_;
    edm::EDGetTokenT<reco::CandidateView> token; 
};

RecoTauPiZeroUnembedder::RecoTauPiZeroUnembedder(const edm::ParameterSet& pset) {
  src_ = pset.getParameter<edm::InputTag>("src");
  token = consumes<reco::CandidateView>(src_);
  produces<reco::RecoTauPiZeroCollection>("pizeros");
  produces<reco::PFTauCollection>();
}
void RecoTauPiZeroUnembedder::produce(edm::Event& evt, const edm::EventSetup& es) {
  std::auto_ptr<reco::RecoTauPiZeroCollection> piZerosOut(
      new reco::RecoTauPiZeroCollection);
  std::auto_ptr<reco::PFTauCollection> tausOut(new reco::PFTauCollection);

  edm::Handle<reco::CandidateView> tauView;
  evt.getByToken(token, tauView);

  reco::PFTauRefVector taus =
      reco::tau::castView<reco::PFTauRefVector>(tauView);

  // Get the reference to the product of where the final pizeros will end up
  reco::RecoTauPiZeroRefProd piZeroProd =
    evt.getRefBeforePut<reco::RecoTauPiZeroCollection>("pizeros");

  for (size_t iTau = 0; iTau < taus.size(); ++iTau) {
    // Make a copy
    reco::PFTau myTau = *taus[iTau];
    // The ref vectors that will be filled
    reco::RecoTauPiZeroRefVector signalPiZeroRefs;
    reco::RecoTauPiZeroRefVector isolationPiZeroRefs;

    // Copy the PiZeros into the new vector, while updating what refs they will
    // have
    const reco::RecoTauPiZeroCollection& signalPiZeros =
      myTau.signalPiZeroCandidates();

    for (size_t iPiZero = 0; iPiZero < signalPiZeros.size(); ++iPiZero) {
      piZerosOut->push_back(signalPiZeros[iPiZero]);
      // Figure out what the ref for this pizero will be in the new coll.
      signalPiZeroRefs.push_back(
          reco::RecoTauPiZeroRef(piZeroProd, piZerosOut->size()-1));
    }

    const reco::RecoTauPiZeroCollection& isolationPiZeroCandidates =
      myTau.isolationPiZeroCandidates();
    for (size_t iPiZero = 0; iPiZero < isolationPiZeroCandidates.size(); ++iPiZero) {
      piZerosOut->push_back(isolationPiZeroCandidates[iPiZero]);
      // Figure out what the ref for this pizero will be in the new coll.
      isolationPiZeroRefs.push_back(
          reco::RecoTauPiZeroRef(piZeroProd, piZerosOut->size()-1));
    }

    myTau.setSignalPiZeroCandidatesRefs(signalPiZeroRefs);
    myTau.setIsolationPiZeroCandidatesRefs(isolationPiZeroRefs);

    tausOut->push_back(myTau);
  }

  evt.put(piZerosOut, "pizeros");
  evt.put(tausOut);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPiZeroUnembedder);
