#include "JetMETCorrections/Type1MET/plugins/PFCandMETcorrInputProducer.h"

#include "DataFormats/Common/interface/View.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

PFCandMETcorrInputProducer::PFCandMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  src_ = cfg.getParameter<edm::InputTag>("src");

  if ( cfg.exists("binning") ) {
    typedef std::vector<edm::ParameterSet> vParameterSet;
    vParameterSet cfgBinning = cfg.getParameter<vParameterSet>("binning");
    for ( vParameterSet::const_iterator cfgBinningEntry = cfgBinning.begin();
	  cfgBinningEntry != cfgBinning.end(); ++cfgBinningEntry ) {
      binning_.push_back(new binningEntryType(*cfgBinningEntry));
    }
  } else {
    binning_.push_back(new binningEntryType());
  }
  
  residualCorrLabel_ = cfg.getParameter<std::string>("residualCorrLabel");
  residualCorrEtaMax_ = cfg.getParameter<double>("residualCorrEtaMax");
  residualCorrOffset_ = cfg.getParameter<double>("residualCorrOffset");
  extraCorrFactor_ = cfg.exists("extraCorrFactor") ? 
    cfg.getParameter<double>("extraCorrFactor") : 1.;

  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    produces<CorrMETData>((*binningEntry)->binLabel_);
  }
}

PFCandMETcorrInputProducer::~PFCandMETcorrInputProducer()
{
  for ( std::vector<binningEntryType*>::const_iterator it = binning_.begin();
	it != binning_.end(); ++it ) {
    delete (*it);
  }
}

void PFCandMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  //std::cout << "<PFCandMETcorrInputProducer::produce>:" << std::endl;

  for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    (*binningEntry)->binUnclEnergySum_ = CorrMETData();
  }

  const JetCorrector* residualCorrector = 0;
  if ( residualCorrLabel_ != "" ) {
    residualCorrector = JetCorrector::getJetCorrector(residualCorrLabel_, es);
    if ( !residualCorrector )  
      throw cms::Exception("PFCandMETcorrInputProducer")
	<< "Failed to access Residual corrections = " << residualCorrLabel_ << " !!\n";
  }
  
  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> pfCandidates;
  evt.getByLabel(src_, pfCandidates);
  
  int pfCandidateIndex = 0;
  for ( edm::View<reco::Candidate>::const_iterator pfCandidate = pfCandidates->begin();
	pfCandidate != pfCandidates->end(); ++pfCandidate ) {
    //std::cout << "PFCandidate #" << pfCandidateIndex << " (raw): Pt = " << pfCandidate->pt() << "," 
    //	        << " eta = " << pfCandidate->eta() << ", phi = " << pfCandidate->phi() << std::endl;
    double residualCorrFactor = 1.;
    if ( residualCorrector && fabs(pfCandidate->eta()) < residualCorrEtaMax_ ) {
      residualCorrFactor = residualCorrector->correction(pfCandidate->p4());
      //std::cout << " residualCorrFactor = " << residualCorrFactor << " (extraCorrFactor = " << extraCorrFactor_ << ")" << std::endl;
    }
    residualCorrFactor *= extraCorrFactor_;
    for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	  binningEntry != binning_.end(); ++binningEntry ) {
      if ( !(*binningEntry)->binSelection_ || (*(*binningEntry)->binSelection_)(pfCandidate->p4()) ) {
	(*binningEntry)->binUnclEnergySum_.mex   += ((residualCorrFactor - residualCorrOffset_)*pfCandidate->px());
	(*binningEntry)->binUnclEnergySum_.mey   += ((residualCorrFactor - residualCorrOffset_)*pfCandidate->py());
	(*binningEntry)->binUnclEnergySum_.sumet += ((residualCorrFactor - residualCorrOffset_)*pfCandidate->et());
      }
    }
    ++pfCandidateIndex;
  }
  
//--- add momentum sum of PFCandidates not within jets ("unclustered energy") to the event
  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*binningEntry)->binUnclEnergySum_)), (*binningEntry)->binLabel_);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFCandMETcorrInputProducer);
