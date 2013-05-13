#include "JetMETCorrections/Type1MET/plugins/PFCandResidualCorrProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/Common/interface/View.h"

#include <math.h>

PFCandResidualCorrProducer::PFCandResidualCorrProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    residualCorrectorFromFile_(0)
{
  src_ = cfg.getParameter<edm::InputTag>("src");
  
  residualCorrLabel_ = cfg.getParameter<std::string>("residualCorrLabel");
  residualCorrEtaMax_ = cfg.getParameter<double>("residualCorrEtaMax");
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

  produces<reco::PFCandidateCollection>();
}

PFCandResidualCorrProducer::~PFCandResidualCorrProducer()
{
  delete residualCorrectorFromFile_;
}

namespace
{
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
  if ( !residualCorrectorFromFile_ ) {
    residualCorrectorFromDB = JetCorrector::getJetCorrector(residualCorrLabel_, es);
    if ( !residualCorrectorFromDB )  
      throw cms::Exception("PFCandResidualCorrProducer")
	<< "Failed to access Residual corrections = " << residualCorrLabel_ << " !!\n";
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
      if ( residualCorrectorFromFile_ ) {
	residualCorrectorFromFile_->setJetEta(pfCandidate->eta());
	residualCorrectorFromFile_->setJetPt(10.);
	residualCorrectorFromFile_->setJetA(0.25);
	residualCorrectorFromFile_->setRho(10.); 
	residualCorrFactor = residualCorrectorFromFile_->getCorrection();
      } else {
	residualCorrFactor = residualCorrectorFromDB->correction(pfCandidate->p4());
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
