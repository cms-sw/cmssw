// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      METAlgo
// 
// Original Authors:  Michael Schmitt, Richard Cavanaugh The University of Florida
//          Created:  May 31, 2005
// $Id: METAlgo.cc,v 1.15 2012/06/08 00:51:28 sakuma Exp $
//

//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <cmath>

CommonMETData METAlgo::run(edm::Handle<edm::View<reco::Candidate> > candidates, double globalThreshold)
{
  CommonMETData met;
  run(candidates, &met, globalThreshold);
  return met;
}

//____________________________________________________________________________||
void METAlgo::run(edm::Handle<edm::View<reco::Candidate> > candidates, CommonMETData *met, double globalThreshold)
{ 
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  double et = 0.0;

  for (unsigned int i = 0; i < candidates->size(); ++i)
  {
    const reco::Candidate &cand = (*candidates)[i];
    if( !(cand.et() > globalThreshold) ) continue;
    px += cand.px();
    py += cand.py();
    pz += cand.pz();
    et += cand.energy()*sin(cand.theta());
  }

  met->mex   = -px;
  met->mey   = -py;

  met->mez   = -pz; // included here since it might be useful
                    // for Data Quality Monitering as it should be 
                    // symmetrically distributed about the origin

  met->met   = sqrt( px*px + py*py );
  met->sumet = et;
  met->phi   = atan2( -py, -px ); // no longer needed as MET is now a candidate
}

//____________________________________________________________________________||
