/*
 * CandViewRefRandomSelector
 *
 * Author: Evan K. Friis (UC Davis)
 *
 * Takes a collection of objects inheriting from Candidates and returns up to
 * N=<choose> candidates.  The N output elements are selected randomly.  If
 * the collection contains N or fewer elements, the entire collection is
 * returned.
 *
 */

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "TRandom3.h"

class CandViewRefRandomSelector : public edm::EDFilter {
  public:
    explicit CandViewRefRandomSelector(const edm::ParameterSet &pset);
    bool filter(edm::Event&, const edm::EventSetup&) override;
  private:
    edm::InputTag src_;
    unsigned int choose_;
    unsigned int seed_;
    bool filter_;
    TRandom3 randy_;
};

CandViewRefRandomSelector::CandViewRefRandomSelector(
    const edm::ParameterSet &pset) {
  src_ = pset.getParameter<edm::InputTag>("src");
  choose_ = pset.getParameter<unsigned int>("choose");
  filter_ = pset.getParameter<bool>("filter");
  seed_ = pset.exists("seed") ? pset.getParameter<unsigned int>("seed") : 123;
  randy_ = TRandom3(seed_);
  produces<reco::CandidateBaseRefVector>();
}

bool CandViewRefRandomSelector::filter(edm::Event& evt,
                                       const edm::EventSetup& es) {
  edm::Handle<edm::View<reco::Candidate> > cands;
  evt.getByLabel(src_, cands);
  std::auto_ptr<reco::CandidateBaseRefVector> output(
      new reco::CandidateBaseRefVector(cands));
  // If we don't have enough elements to select, just copy what we have
  if (cands->size() <= choose_) {
    for (size_t i = 0; i < cands->size(); ++i)
        output->push_back(cands->refAt(i));
  } else {
    for (size_t i = 0; i < cands->size() && output->size() < choose_; ++i) {
      // based on http://stackoverflow.com/questions/48087/
      // select-a-random-n-elements-from-listt-in-c/48089#48089
      double selectionProb =
          (choose_ - output->size())*1.0/(cands->size() - i);
      // throw a number to see if we select this element
      if (randy_.Rndm() < selectionProb)
        output->push_back(cands->refAt(i));
    }
  }
  size_t outputSize = output->size();
  evt.put(output);
  return ( !filter_ || outputSize );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandViewRefRandomSelector);
