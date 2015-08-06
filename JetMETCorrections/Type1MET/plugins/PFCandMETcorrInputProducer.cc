#include "JetMETCorrections/Type1MET/plugins/PFCandMETcorrInputProducer.h"

#include "DataFormats/Common/interface/View.h"

PFCandMETcorrInputProducer::PFCandMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  token_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("src"));

  if ( cfg.exists("binning") ) {
    typedef std::vector<edm::ParameterSet> vParameterSet;
    vParameterSet cfgBinning = cfg.getParameter<vParameterSet>("binning");
    for ( vParameterSet::const_iterator cfgBinningEntry = cfgBinning.begin();
	  cfgBinningEntry != cfgBinning.end(); ++cfgBinningEntry ) {
      binning_.emplace_back(new binningEntryType(*cfgBinningEntry));
    }
  } else {
    binning_.emplace_back(new binningEntryType());
  }
  
  for ( auto binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    produces<CorrMETData>((*binningEntry)->binLabel_);
  }
}

PFCandMETcorrInputProducer::~PFCandMETcorrInputProducer()
{  
}

void PFCandMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  for ( auto binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    (*binningEntry)->binUnclEnergySum_ = CorrMETData();
  }

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> cands;
  evt.getByToken(token_, cands);
  
  for ( edm::View<reco::Candidate>::const_iterator cand = cands->begin();
	cand != cands->end(); ++cand ) {
    for ( auto binningEntry = binning_.begin();
	  binningEntry != binning_.end(); ++binningEntry ) {
      if ( !(*binningEntry)->binSelection_ || (*(*binningEntry)->binSelection_)(cand->p4()) ) {
	(*binningEntry)->binUnclEnergySum_.mex   += cand->px();
	(*binningEntry)->binUnclEnergySum_.mey   += cand->py();
	(*binningEntry)->binUnclEnergySum_.sumet += cand->et();
      }
    }
  }

//--- add momentum sum of PFCandidates not within jets ("unclustered energy") to the event
  for ( auto binningEntry = binning_.cbegin();
	binningEntry != binning_.cend(); ++binningEntry ) {
    evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*binningEntry)->binUnclEnergySum_)), (*binningEntry)->binLabel_);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFCandMETcorrInputProducer);
