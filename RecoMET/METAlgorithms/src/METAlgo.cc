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
CommonMETData METAlgo::run(const edm::View<reco::Candidate>& candidates, double globalThreshold)
{
  math::XYZTLorentzVector p4;
  for(auto cand = candidates.begin(); cand != candidates.end(); ++cand)
    {
      if( !(cand->et() > globalThreshold) ) continue;
      p4 += cand->p4();
    }
  math::XYZTLorentzVector met = -p4;


  CommonMETData ret;
  ret.mex   = met.Px();
  ret.mey   = met.Py();

  ret.mez   = met.Pz(); // included here since it might be useful
                        // for Data Quality Monitering as it should be
                        // symmetrically distributed about the origin

  ret.met   = met.Pt();
  ret.phi   = met.Phi(); // no longer needed as MET is now a candidate


  double et = 0.0;
  for(auto cand = candidates.begin(); cand != candidates.end(); ++cand)
    {
      if( !(cand->et() > globalThreshold) ) continue;
      et += cand->et();
    }

  ret.sumet = et;

  return ret;
}

//____________________________________________________________________________||
