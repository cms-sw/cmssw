#ifndef PhysicsTools_PatUtils_PFIsolation_h
#define PhysicsTools_PatUtils_PFIsolation_h

/*
  Defines a function to compute MiniIsolation given a 4-vector and a collection
  of packed PF candidates.

  Mini-Isolation reference: https://hypernews.cern.ch/HyperNews/CMS/get/susy/1991.html
*/

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace pat{

  float miniIsoDr(const math::XYZTLorentzVector &p4, float mindr, float maxdr,
		  float kt_scale);

  // see src file for definitions of parameters
  PFIsolation getMiniPFIsolation(const pat::PackedCandidateCollection *pfcands, const math::XYZTLorentzVector& p4,
                                   float mindr=0.05, float maxdr=0.2, float kt_scale=10.0,
                                   float ptthresh=0.5, float deadcone_ch=0.0001,
                                   float deadcone_pu=0.01, float deadcone_ph=0.01, float deadcone_nh=0.01,
                                   float dZ_cut=0.0);

  float muonRelMiniIsoPUCorrected(const PFIsolation& iso,
				  const math::XYZTLorentzVector& p4,
				  float dr,
				  float rho);
}

#endif
