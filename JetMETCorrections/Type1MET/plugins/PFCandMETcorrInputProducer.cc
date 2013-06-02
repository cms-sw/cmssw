#include "JetMETCorrections/Type1MET/plugins/PFCandMETcorrInputProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/Common/interface/View.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

namespace
{
  FactorizedJetCorrector* getResidualCorrector(const edm::FileInPath& residualCorrFileName)
  {
    FactorizedJetCorrector* residualCorrector = 0;
    if ( !residualCorrFileName.isLocal() ) 
      throw cms::Exception("PFCandMETcorrInputProducer") 
	<< " Failed to find File = " << residualCorrFileName << " !!\n";
    JetCorrectorParameters residualCorr(residualCorrFileName.fullPath().data());
    std::vector<JetCorrectorParameters> jetCorrections;
    jetCorrections.push_back(residualCorr);
    residualCorrector = new FactorizedJetCorrector(jetCorrections);
    return residualCorrector;
  }  
}

PFCandMETcorrInputProducer::PFCandMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    residualCorrectorFromFile_(0),
    residualCorrectorVsNumPileUp_data_offset_(0),
    residualCorrectorVsNumPileUp_data_slope_(0),
    residualCorrectorVsNumPileUp_mc_offset_(0),
    residualCorrectorVsNumPileUp_mc_slope_(0),
    mode_(kResidualCorrFromDB)
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
    residualCorrectorFromFile_ = getResidualCorrector(residualCorrFileName);
    mode_ = kResidualCorrFromFile;
  }

  isMC_ = cfg.getParameter<bool>("isMC");

  if ( cfg.exists("residualCorrVsNumPileUp") ) {
    if ( !isMC_ ) 
      throw cms::Exception("PFCandMETcorrInputProducer")
	<< "Pile-up dependent Residual corrections must be applied to Monte Carlo only !!\n";
    srcGenPileUpSummary_ = cfg.getParameter<edm::InputTag>("srcGenPileUpSummary");
    edm::ParameterSet cfgResidualCorrVsNumPileUp = cfg.getParameter<edm::ParameterSet>("residualCorrVsNumPileUp");
    edm::ParameterSet cfgResidualCorr_data = cfgResidualCorrVsNumPileUp.getParameter<edm::ParameterSet>("data");
    residualCorrectorVsNumPileUp_data_offset_ = getResidualCorrector(cfgResidualCorr_data.getParameter<edm::FileInPath>("offset"));
    residualCorrectorVsNumPileUp_data_slope_ = getResidualCorrector(cfgResidualCorr_data.getParameter<edm::FileInPath>("slope"));
    edm::ParameterSet cfgResidualCorr_mc = cfgResidualCorrVsNumPileUp.getParameter<edm::ParameterSet>("mc");
    residualCorrectorVsNumPileUp_mc_offset_ = getResidualCorrector(cfgResidualCorr_mc.getParameter<edm::FileInPath>("offset"));
    residualCorrectorVsNumPileUp_mc_slope_ = getResidualCorrector(cfgResidualCorr_mc.getParameter<edm::FileInPath>("slope"));
    mode_ = kResidualCorrVsNumPileUp;
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  if ( verbosity_ ) {
    std::cout << "<PFCandMETcorrInputProducer::PFCandMETcorrInputProducer>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
    if      ( mode_ == kResidualCorrFromDB      ) std::cout << "applying Residual correction = " << residualCorrLabel_ << "*" << extraCorrFactor_ << " from DataBase." << std::endl;
    else if ( mode_ == kResidualCorrFromFile    ) std::cout << "applying Residual correction = " << residualCorrLabel_ << "*" << extraCorrFactor_ << " from File." << std::endl;
    else if ( mode_ == kResidualCorrVsNumPileUp ) std::cout << "applying Pile-up dependent Residual corrections." << std::endl;
  }

  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    produces<CorrMETData>((*binningEntry)->binLabel_);
  }
}

PFCandMETcorrInputProducer::~PFCandMETcorrInputProducer()
{
  delete residualCorrectorFromFile_;

  delete residualCorrectorVsNumPileUp_data_offset_;
  delete residualCorrectorVsNumPileUp_data_slope_;
  delete residualCorrectorVsNumPileUp_mc_offset_;
  delete residualCorrectorVsNumPileUp_mc_slope_;

  for ( std::vector<binningEntryType*>::const_iterator it = binning_.begin();
	it != binning_.end(); ++it ) {
    delete (*it);
  }
}

namespace
{
  double getResidualCorrection(FactorizedJetCorrector* residualCorrector, double eta)
  {
    residualCorrector->setJetEta(eta);
    residualCorrector->setJetPt(10.);
    residualCorrector->setJetA(0.25);
    residualCorrector->setRho(10.); 
    double residualCorrection = residualCorrector->getCorrection();
    return residualCorrection;
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
  if ( mode_ == kResidualCorrFromDB && residualCorrLabel_ != "" ) {
    residualCorrectorFromDB = JetCorrector::getJetCorrector(residualCorrLabel_, es);
    if ( !residualCorrectorFromDB )  
      throw cms::Exception("PFCandMETcorrInputProducer")
	<< "Failed to access Residual corrections = " << residualCorrLabel_ << " !!\n";
  }

  double numPileUp = -1;
  if ( mode_ == kResidualCorrVsNumPileUp ) {
    typedef std::vector<PileupSummaryInfo> PileupSummaryInfoCollection;
    edm::Handle<PileupSummaryInfoCollection> genPileUpInfos;
    evt.getByLabel(srcGenPileUpSummary_, genPileUpInfos);
    for ( PileupSummaryInfoCollection::const_iterator genPileUpInfo = genPileUpInfos->begin();
	  genPileUpInfo != genPileUpInfos->end(); ++genPileUpInfo ) {
      // CV: in-time PU is stored in getBunchCrossing = 0, 
      //    cf. https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupInformation
      int bx = genPileUpInfo->getBunchCrossing();
      if ( verbosity_ ) {
	std::cout << "bx = " << bx << ": numPileUpInteractions = " << genPileUpInfo->getPU_NumInteractions() << " (true = " << genPileUpInfo->getTrueNumInteractions() << ")" << std::endl;
      }
      if ( bx == 0 ) {
	numPileUp = genPileUpInfo->getTrueNumInteractions();
      }
    }
    if ( numPileUp == -1. ) 
      throw cms::Exception("PFCandMETcorrInputProducer")
	<< " Failed to decode in-time Pile-up information stored in PileupSummaryInfo object !!\n";
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
      if ( mode_ == kResidualCorrFromFile && residualCorrLabel_ != "" ) {
	residualCorrFactor = getResidualCorrection(residualCorrectorFromFile_, pfCandidate->eta());
      } else if ( mode_ == kResidualCorrFromDB && residualCorrLabel_ != "" ) {
	residualCorrFactor = residualCorrectorFromDB->correction(pfCandidate->p4());
      } else if ( mode_ == kResidualCorrVsNumPileUp ) {
	double residualCorrParameter_data_offset = getResidualCorrection(residualCorrectorVsNumPileUp_data_offset_, pfCandidate->eta());
	double residualCorrParameter_data_slope  = getResidualCorrection(residualCorrectorVsNumPileUp_data_slope_, pfCandidate->eta());
	double response_data = residualCorrParameter_data_offset + residualCorrParameter_data_slope*numPileUp;
	double residualCorrParameter_mc_offset   = getResidualCorrection(residualCorrectorVsNumPileUp_mc_offset_, pfCandidate->eta());
	double residualCorrParameter_mc_slope    = getResidualCorrection(residualCorrectorVsNumPileUp_mc_slope_, pfCandidate->eta());
	double response_mc = residualCorrParameter_mc_offset + residualCorrParameter_mc_slope*numPileUp;
	if ( verbosity_ ) std::cout << "response(eta = " << pfCandidate->eta() << "): data = " << response_data << ", mc = " << response_mc << std::endl;
	if ( response_data > 0. ) {
	  residualCorrFactor = response_mc/response_data;
	}	    
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
