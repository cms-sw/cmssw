#include "JetMETCorrections/Type1MET/plugins/PFCandResidualCorrProducer.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/Common/interface/View.h"

#include <math.h>

PFCandResidualCorrProducer::PFCandResidualCorrProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  src_ = cfg.getParameter<edm::InputTag>("src");
  
  residualCorrLabel_ = cfg.getParameter<std::string>("residualCorrLabel");
  residualCorrEtaMax_ = cfg.getParameter<double>("residualCorrEtaMax");
  extraCorrFactor_ = cfg.exists("extraCorrFactor") ? 
    cfg.getParameter<double>("extraCorrFactor") : 1.;

  produces<reco::PFCandidateCollection>();
}

PFCandResidualCorrProducer::~PFCandResidualCorrProducer()
{}

namespace
{
  double square(double x)
  {
    return x*x;
  }
}

void PFCandResidualCorrProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  //std::cout << "<PFCandResidualCorrProducer::produce>:" << std::endl;

  std::auto_ptr<reco::PFCandidateCollection> pfCandidates_corrected(new reco::PFCandidateCollection());

  const JetCorrector* residualCorrector = JetCorrector::getJetCorrector(residualCorrLabel_, es);
  if ( !residualCorrector )  
    throw cms::Exception("PFCandResidualCorrProducer")
      << "Failed to access Residual corrections = " << residualCorrLabel_ << " !!\n";

  typedef edm::View<reco::PFCandidate> PFCandidateView;
  edm::Handle<PFCandidateView> pfCandidates;
  evt.getByLabel(src_, pfCandidates);
  int pfCandidateIndex = 0;
  for ( PFCandidateView::const_iterator pfCandidate = pfCandidates->begin();
	pfCandidate != pfCandidates->end(); ++pfCandidate ) {
    //std::cout << "PFCandidate #" << pfCandidateIndex << " (raw): Pt = " << pfCandidate->pt() << "," 
    //	        << " eta = " << pfCandidate->eta() << ", phi = " << pfCandidate->phi() << std::endl;
    double residualCorrFactor = 1.;
    if ( fabs(pfCandidate->eta()) < residualCorrEtaMax_ ) {
      residualCorrFactor = residualCorrector->correction(pfCandidate->p4());
      //std::cout << " residualCorrFactor = " << residualCorrFactor << " (extraCorrFactor = " << extraCorrFactor_ << ")" << std::endl;
    }
    residualCorrFactor *= extraCorrFactor_;
    double pfCandidatePx_corrected = residualCorrFactor*pfCandidate->px();
    double pfCandidatePy_corrected = residualCorrFactor*pfCandidate->py();
    double pfCandidatePz_corrected = residualCorrFactor*pfCandidate->pz();
    double pfCandidateEn_corrected = sqrt(square(pfCandidatePx_corrected) + square(pfCandidatePy_corrected) + square(pfCandidatePz_corrected) + square(pfCandidate->mass()));
    reco::Candidate::LorentzVector pfCandidateP4_corrected(pfCandidatePx_corrected, pfCandidatePy_corrected, pfCandidatePz_corrected, pfCandidateEn_corrected);
    reco::PFCandidate pfCandidate_corrected(*pfCandidate);
    //std::cout << "PFCandidate #" << pfCandidateIndex << " (corrected): Pt = " << pfCandidate_corrected.pt() << "," 
    //	        << " eta = " << pfCandidate_corrected.eta() << ", phi = " << pfCandidate_corrected.phi() << std::endl;
    pfCandidate_corrected.setP4(pfCandidateP4_corrected);
    pfCandidates_corrected->push_back(pfCandidate_corrected);
    ++pfCandidateIndex;
  }

  evt.put(pfCandidates_corrected);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFCandResidualCorrProducer);
