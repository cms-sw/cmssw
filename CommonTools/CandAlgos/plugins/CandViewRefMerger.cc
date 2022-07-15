/* \class CandViewRefMerger
 *
 * Producer of merged references to Candidates
 *
 * \author: Luca Lista, INFN
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/Common/interface/Handle.h"

class CandViewRefMerger : public edm::global::EDProducer<> {
public:
  explicit CandViewRefMerger(const edm::ParameterSet& cfg)
      : srcTokens_(
            edm::vector_transform(cfg.getParameter<std::vector<edm::InputTag> >("src"),
                                  [this](edm::InputTag const& tag) { return consumes<reco::CandidateView>(tag); })) {
    produces<std::vector<reco::CandidateBaseRef> >();
  }

private:
  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override {
    std::unique_ptr<std::vector<reco::CandidateBaseRef> > out(new std::vector<reco::CandidateBaseRef>);
    for (std::vector<edm::EDGetTokenT<reco::CandidateView> >::const_iterator i = srcTokens_.begin();
         i != srcTokens_.end();
         ++i) {
      edm::Handle<reco::CandidateView> src;
      evt.getByToken(*i, src);
      for (size_t j = 0; j < src->size(); ++j)
        out->push_back(src->refAt(j));
    }
    evt.put(std::move(out));
  }
  std::vector<edm::EDGetTokenT<reco::CandidateView> > srcTokens_;
};

DEFINE_FWK_MODULE(CandViewRefMerger);
