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
  // define a float-precision version of the typedefs in reco::PFTauTransverseImpactParameter class
  typedef math::PtEtaPhiMLorentzVectorF LorentzVector;
  typedef math::XYZPointF Point;
  typedef math::XYZVectorF Vector;
  typedef math::ErrorF<3>::type CovMatrix;

// dummy constructor for ROOT I/O
  TauPFEssential() {}
// constructor from PFTau
  TauPFEssential(const reco::PFTau& tau);
// datamembers 
  LorentzVector p4Jet_;
  LorentzVector p4CorrJet_;
  
  int decayMode_;
  
  Point dxy_PCA_;
  float dxy_;
  float dxy_error_;
  float dxy_Sig_;
  reco::VertexRef pv_;
  Point pvPos_;
  CovMatrix pvCov_;
  bool hasSV_;
  Vector flightLength_;
  float flightLengthSig_;
  reco::VertexRef sv_;
  Point svPos_;
  CovMatrix svCov_;
};

} }

#endif
