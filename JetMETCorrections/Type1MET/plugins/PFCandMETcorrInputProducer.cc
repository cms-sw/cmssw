#include "JetMETCorrections/Type1MET/plugins/PFCandMETcorrInputProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/View.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

PFCandMETcorrInputProducer::PFCandMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    residualCorrectorFromFile_(0)
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
  if ( cfg.exists("residualCorrFileName") ) {
    edm::FileInPath residualCorrFileName = cfg.getParameter<edm::FileInPath>("residualCorrFileName");
    if ( !residualCorrFileName.isLocal()) 
      throw cms::Exception("calibUnclusteredEnergy") 
	<< " Failed to find File = " << residualCorrFileName << " !!\n";
    JetCorrectorParameters residualCorr(residualCorrFileName.fullPath().data());
    std::vector<JetCorrectorParameters> jetCorrections;
    jetCorrections.push_back(residualCorr);
    residualCorrectorFromFile_ = new FactorizedJetCorrector(jetCorrections);
  }  
  isMC_ = cfg.getParameter<bool>("isMC");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    produces<CorrMETData>((*binningEntry)->binLabel_);
  }
}

PFCandMETcorrInputProducer::~PFCandMETcorrInputProducer()
{
  delete residualCorrectorFromFile_;
  for ( std::vector<binningEntryType*>::const_iterator it = binning_.begin();
	it != binning_.end(); ++it ) {
    delete (*it);
  }
}

void PFCandMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) {
    std::cout << "<PFCandMETcorrInputProducer::produce>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  }

  for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    (*binningEntry)->binUnclEnergySum_ = CorrMETData();
  }

  const JetCorrector* residualCorrectorFromDB = 0;
  if ( !residualCorrectorFromFile_ && residualCorrLabel_ != "" ) {
    residualCorrectorFromDB = JetCorrector::getJetCorrector(residualCorrLabel_, es);
    if ( !residualCorrectorFromDB )  
      throw cms::Exception("PFCandMETcorrInputProducer")
	<< "Failed to access Residual corrections = " << residualCorrLabel_ << " !!\n";
  }
  
  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> pfCandidates;
  evt.getByLabel(src_, pfCandidates);
  if ( verbosity_ ) { 
    std::cout << "#pfCandidates = " << pfCandidates->size() << std::endl;
  }
  
  int pfCandidateIndex = 0;
  for ( edm::View<reco::Candidate>::const_iterator pfCandidate = pfCandidates->begin();
	pfCandidate != pfCandidates->end(); ++pfCandidate ) {
    if ( verbosity_ ) { 
      std::cout << "PFCandidate #" << pfCandidateIndex << " (raw): Pt = " << pfCandidate->pt() << "," 
		<< " eta = " << pfCandidate->eta() << ", phi = " << pfCandidate->phi() << std::endl;
    }

    double residualCorrFactor = 1.;
    if ( fabs(pfCandidate->eta()) < residualCorrEtaMax_ ) {
      if ( residualCorrectorFromFile_ ) {
	residualCorrectorFromFile_->setJetEta(pfCandidate->eta());
	residualCorrectorFromFile_->setJetPt(10.);
	residualCorrectorFromFile_->setJetA(0.25);
	residualCorrectorFromFile_->setRho(10.); 
	residualCorrFactor = residualCorrectorFromFile_->getCorrection();
      } else if ( residualCorrectorFromDB ) {	
	residualCorrFactor = residualCorrectorFromDB->correction(pfCandidate->p4());
      }
      if ( verbosity_ ) std::cout << " residualCorrFactor = " << residualCorrFactor << " (extraCorrFactor = " << extraCorrFactor_ << ")" << std::endl;
    }
    residualCorrFactor *= extraCorrFactor_;
    if ( isMC_ && residualCorrFactor != 0. ) residualCorrFactor = 1./residualCorrFactor;

    for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	  binningEntry != binning_.end(); ++binningEntry ) {
      if ( !(*binningEntry)->binSelection_ || (*(*binningEntry)->binSelection_)(pfCandidate->p4()) ) {
	if ( verbosity_ ) std::cout << "adding PFCandidate." << std::endl;
	(*binningEntry)->binUnclEnergySum_.mex   += ((residualCorrFactor - residualCorrOffset_)*pfCandidate->px());
	(*binningEntry)->binUnclEnergySum_.mey   += ((residualCorrFactor - residualCorrOffset_)*pfCandidate->py());
	(*binningEntry)->binUnclEnergySum_.sumet += ((residualCorrFactor - residualCorrOffset_)*pfCandidate->et());
      }
    }
    ++pfCandidateIndex;
  }

  if ( verbosity_ ) { 
    for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	  binningEntry != binning_.end(); ++binningEntry ) {
      std::cout << (*binningEntry)->binLabel_ << ":" 
		<< " Px = " << (*binningEntry)->binUnclEnergySum_.mex << "," 
		<< " Py = " << (*binningEntry)->binUnclEnergySum_.mey << "," 
		<< " sumEt = " << (*binningEntry)->binUnclEnergySum_.sumet << std::endl;
    }
  }
  
//--- add momentum sum of PFCandidates not within jets ("unclustered energy") to the event
  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*binningEntry)->binUnclEnergySum_)), (*binningEntry)->binLabel_);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFCandMETcorrInputProducer);
