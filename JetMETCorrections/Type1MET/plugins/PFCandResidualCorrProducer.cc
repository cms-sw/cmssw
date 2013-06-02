#include "JetMETCorrections/Type1MET/plugins/PFCandResidualCorrProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/Common/interface/View.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <math.h>

namespace
{
  FactorizedJetCorrector* getResidualCorrector(const edm::FileInPath& residualCorrFileName)
  {
    FactorizedJetCorrector* residualCorrector = 0;
    if ( !residualCorrFileName.isLocal() ) 
      throw cms::Exception("PFCandResidualCorrProducer") 
	<< " Failed to find File = " << residualCorrFileName << " !!\n";
    JetCorrectorParameters residualCorr(residualCorrFileName.fullPath().data());
    std::vector<JetCorrectorParameters> jetCorrections;
    jetCorrections.push_back(residualCorr);
    residualCorrector = new FactorizedJetCorrector(jetCorrections);
    return residualCorrector;
  }  
}

PFCandResidualCorrProducer::PFCandResidualCorrProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    residualCorrectorFromFile_(0),
    residualCorrectorVsNumPileUp_data_offset_(0),
    residualCorrectorVsNumPileUp_data_slope_(0),
    residualCorrectorVsNumPileUp_mc_offset_(0),
    residualCorrectorVsNumPileUp_mc_slope_(0),
    mode_(kResidualCorrFromDB)
{
  src_ = cfg.getParameter<edm::InputTag>("src");
  
  residualCorrLabel_ = cfg.getParameter<std::string>("residualCorrLabel");
  residualCorrEtaMax_ = cfg.getParameter<double>("residualCorrEtaMax");
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
      throw cms::Exception("PFCandResidualCorrProducer")
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
    std::cout << "<PFCandResidualCorrProducer::PFCandResidualCorrProducer>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
    if      ( mode_ == kResidualCorrFromDB      ) std::cout << "applying Residual correction = " << residualCorrLabel_ << "*" << extraCorrFactor_ << " from DataBase." << std::endl;
    else if ( mode_ == kResidualCorrFromFile    ) std::cout << "applying Residual correction = " << residualCorrLabel_ << "*" << extraCorrFactor_ << " from File." << std::endl;
    else if ( mode_ == kResidualCorrVsNumPileUp ) std::cout << "applying Pile-up dependent Residual corrections." << std::endl;
  }

  produces<reco::PFCandidateCollection>();
}

PFCandResidualCorrProducer::~PFCandResidualCorrProducer()
{
  delete residualCorrectorFromFile_;

  delete residualCorrectorVsNumPileUp_data_offset_;
  delete residualCorrectorVsNumPileUp_data_slope_;
  delete residualCorrectorVsNumPileUp_mc_offset_;
  delete residualCorrectorVsNumPileUp_mc_slope_;
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

  double square(double x)
  {
    return x*x;
  }
}

void PFCandResidualCorrProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) {
    std::cout << "<PFCandResidualCorrProducer::produce>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  }

  std::auto_ptr<reco::PFCandidateCollection> pfCandidates_corrected(new reco::PFCandidateCollection());

  const JetCorrector* residualCorrectorFromDB = 0;
  if ( mode_ == kResidualCorrFromDB && residualCorrLabel_ != "" ) {
    residualCorrectorFromDB = JetCorrector::getJetCorrector(residualCorrLabel_, es);
    if ( !residualCorrectorFromDB )  
      throw cms::Exception("PFCandResidualCorrProducer")
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
      throw cms::Exception("PFCandResidualCorrProducer")
	<< " Failed to decode in-time Pile-up information stored in PileupSummaryInfo object !!\n";
  }
  
  typedef edm::View<reco::PFCandidate> PFCandidateView;
  edm::Handle<PFCandidateView> pfCandidates;
  evt.getByLabel(src_, pfCandidates);
  int pfCandidateIndex = 0;
  for ( PFCandidateView::const_iterator pfCandidate = pfCandidates->begin();
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

    double pfCandidatePx_corrected = residualCorrFactor*pfCandidate->px();
    double pfCandidatePy_corrected = residualCorrFactor*pfCandidate->py();
    double pfCandidatePz_corrected = residualCorrFactor*pfCandidate->pz();
    double pfCandidateEn_corrected = sqrt(square(pfCandidatePx_corrected) + square(pfCandidatePy_corrected) + square(pfCandidatePz_corrected) + square(pfCandidate->mass()));
    reco::Candidate::LorentzVector pfCandidateP4_corrected(pfCandidatePx_corrected, pfCandidatePy_corrected, pfCandidatePz_corrected, pfCandidateEn_corrected);
    reco::PFCandidate pfCandidate_corrected(*pfCandidate);
    if ( verbosity_ ) {
      std::cout << "PFCandidate #" << pfCandidateIndex << " (corrected): Pt = " << pfCandidate_corrected.pt() << "," 
    	        << " eta = " << pfCandidate_corrected.eta() << ", phi = " << pfCandidate_corrected.phi() << std::endl;
    }
    pfCandidate_corrected.setP4(pfCandidateP4_corrected);
    pfCandidates_corrected->push_back(pfCandidate_corrected);
    ++pfCandidateIndex;
  }

  evt.put(pfCandidates_corrected);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFCandResidualCorrProducer);
