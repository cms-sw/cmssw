#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace edm;
using namespace reco;

class CandCollectionExistFilter : public EDFilter {
public: 
  CandCollectionExistFilter(const ParameterSet & cfg) :
    src_(cfg.getParameter<InputTag>("src")) { }
private:
  bool filter(Event& evt, const EventSetup&) {
    Handle<CandidateView> src;
    bool exists = true;
    evt.getByLabel(src_, src);
    if(!src.isValid()) exists = false;
    return exists;
  }
  InputTag src_;
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CandCollectionExistFilter);

