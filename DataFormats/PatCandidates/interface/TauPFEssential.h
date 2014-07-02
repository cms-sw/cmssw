//
//

#ifndef DataFormats_PatCandidates_Tau_PFEssential_h
#define DataFormats_PatCandidates_Tau_PFEssential_h

/**
  \class    pat::tau::PFEssential TauPFEssential.h "DataFormats/PatCandidates/interface/TauPFEssential.h"
  \brief    Structure to hold information from PFTau essential for analysis inside a pat::Tau

  \author   Pavel Jez
*/

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace pat { namespace tau {

struct TauPFEssential {
// dummy constructor for ROOT I/O
  TauPFEssential() {}
// constructor from PFTau
  TauPFEssential(const reco::PFTau& tau);
// datamembers 
  reco::Candidate::LorentzVector p4Jet_;
  reco::Candidate::LorentzVector p4CorrJet_;
  
  int decayMode_;
  
  reco::PFTauTransverseImpactParameter::Point dxy_PCA_;
  double dxy_;
  double dxy_error_;
  double dxy_Sig_;
  reco::VertexRef pv_;
  reco::PFTauTransverseImpactParameter::Point pvPos_;
  reco::PFTauTransverseImpactParameter::CovMatrix pvCov_;
  bool hasSV_;
  reco::PFTauTransverseImpactParameter::Vector flightLength_;
  double flightLengthSig_;
  reco::VertexRef sv_;
  reco::PFTauTransverseImpactParameter::Point svPos_;
  reco::PFTauTransverseImpactParameter::CovMatrix svCov_;
};

} }

#endif
