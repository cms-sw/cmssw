/*
 * CandViewRefTriggerBiasRemover
 *
 * Author: Evan K. Friis
 *
 * Takes a collection of "triggered" objects and returns produces a collection
 * that removes all elements that are biased by the trigger.  In practice, this
 * returns an empty colleciton if the size of the input collection is 1, and the
 * entire collection if the input collection has at least two elements.
 *
 * In summary, for any element in the output collection, there exists at least
 * one *other* element in the output collection that fired the trigger.
 *
 */

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

class CandViewRefTriggerBiasRemover : public edm::EDProducer {
  public:
    CandViewRefTriggerBiasRemover(const edm::ParameterSet &pset);
    void produce(edm::Event&, const edm::EventSetup&) override;
  private:
    edm::InputTag src_;
};

CandViewRefTriggerBiasRemover::CandViewRefTriggerBiasRemover(
    const edm::ParameterSet &pset) {
  src_ = pset.getParameter<edm::InputTag>("triggered");
  produces<reco::CandidateBaseRefVector>();
}

void CandViewRefTriggerBiasRemover::produce(edm::Event& evt,
                                            const edm::EventSetup& es) {
  edm::Handle<edm::View<reco::Candidate> > cands;
  evt.getByLabel(src_, cands);
  std::auto_ptr<reco::CandidateBaseRefVector> output(
      new reco::CandidateBaseRefVector(cands));
  // Only copy the output if there is more than one item in the input
  size_t nCands = cands->size();
  if (nCands > 1) {
    //output->reserve(nCands);
    for (size_t iCand = 0; iCand < nCands; ++iCand) {
      output->push_back(cands->refAt(iCand));
    }
  }
  evt.put(output);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandViewRefTriggerBiasRemover);
