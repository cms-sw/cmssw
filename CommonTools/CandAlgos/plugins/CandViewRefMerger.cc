/* \class CandViewRefMerger
 * 
 * Producer of merged references to Candidates
 *
 * \author: Luca Lista, INFN
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

class CandViewRefMerger : public edm::EDProducer {
public:
  explicit CandViewRefMerger(const edm::ParameterSet& cfg) :
    src_(cfg.getParameter<std::vector<edm::InputTag> >("src")) {
    produces<std::vector<reco::CandidateBaseRef> >();
  }
private:
  void produce(edm::Event & evt, const edm::EventSetup &) {
    std::auto_ptr<std::vector<reco::CandidateBaseRef> > out(new std::vector<reco::CandidateBaseRef>);
    for(std::vector<edm::InputTag>::const_iterator i = src_.begin(); i != src_.end(); ++i) {
      edm::Handle<reco::CandidateView> src;
      evt.getByLabel(*i, src);
      reco::CandidateBaseRefVector refs = src->refVector();
      for(reco::CandidateBaseRefVector::const_iterator j = refs.begin(); j != refs.end(); ++j)
	out->push_back(*j);
    }
    evt.put(out);
  }
  std::vector<edm::InputTag> src_;
};

DEFINE_FWK_MODULE(CandViewRefMerger);
