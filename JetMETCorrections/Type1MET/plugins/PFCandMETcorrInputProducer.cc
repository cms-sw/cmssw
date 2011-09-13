#include "JetMETCorrections/Type1MET/plugins/PFCandMETcorrInputProducer.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "JetMETCorrections/Type1MET/interface/metCorrAuxFunctions.h"

using namespace metCorr_namespace;

PFCandMETcorrInputProducer::PFCandMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  src_ = cfg.getParameter<edm::InputTag>("src");

  produces<CorrMETData>();
}

PFCandMETcorrInputProducer::~PFCandMETcorrInputProducer()
{
// nothing to be done yet...
}

void PFCandMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<CorrMETData> unclEnergySum(new CorrMETData());

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> cands;
  evt.getByLabel(src_, cands);

  for ( edm::View<reco::Candidate>::const_iterator cand = cands->begin();
	cand != cands->end(); ++cand ) {
    unclEnergySum->mex   += cand->px();
    unclEnergySum->mey   += cand->py();
    unclEnergySum->sumet += cand->et();
  }

//--- add momentum sum of PFCandidates not within jets ("unclustered energy") to the event
  evt.put(unclEnergySum);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFCandMETcorrInputProducer);
