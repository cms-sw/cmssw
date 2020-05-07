// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      METAlgo
//
// Original Authors:  Michael Schmitt, Richard Cavanaugh The University of Florida
//          Created:  May 31, 2005
//

//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"

//____________________________________________________________________________||
CommonMETData METAlgo::run(const edm::View<reco::Candidate>& candidates,
                           double globalThreshold,
                           edm::ValueMap<float> const* weights) {
  math::XYZTLorentzVector p4;
  for (auto const& candPtr : candidates.ptrs()) {
    const reco::Candidate* cand = candPtr.get();
    float weight = (weights != nullptr) ? (*weights)[candPtr] : 1.0;
    if (!(cand->et() * weight > globalThreshold))
      continue;
    p4 += cand->p4() * weight;
  }
  math::XYZTLorentzVector met = -p4;

  CommonMETData ret;
  ret.mex = met.Px();
  ret.mey = met.Py();

  ret.mez = met.Pz();  // included here since it might be useful
                       // for Data Quality Monitering as it should be
                       // symmetrically distributed about the origin

  ret.met = met.Pt();

  double et = 0.0;
  for (auto const& candPtr : candidates.ptrs()) {
    const reco::Candidate* cand = candPtr.get();
    float weight = (weights != nullptr) ? (*weights)[candPtr] : 1.0;
    if (!(cand->et() * weight > globalThreshold))
      continue;
    et += cand->et() * weight;
  }

  ret.sumet = et;

  return ret;
}

//____________________________________________________________________________||
