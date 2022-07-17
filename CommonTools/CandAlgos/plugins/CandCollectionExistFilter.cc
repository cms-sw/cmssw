#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace edm;
using namespace reco;

class CandCollectionExistFilter : public edm::stream::EDFilter<> {
public:
  CandCollectionExistFilter(const ParameterSet& cfg)
      : srcToken_(consumes<CandidateView>(cfg.getParameter<InputTag>("src"))) {}

private:
  bool filter(Event& evt, const EventSetup&) override { return evt.getHandle(srcToken_).isValid(); }
  const EDGetTokenT<CandidateView> srcToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandCollectionExistFilter);
