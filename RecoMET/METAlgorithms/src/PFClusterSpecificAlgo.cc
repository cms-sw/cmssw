// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      PFClusterSpecificAlgo
// 
// Original Authors:  R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
//          Created:  October 27, 2008
// $Id: PFClusterSpecificAlgo.cc,v 1.4 2012/06/11 04:27:43 sakuma Exp $
//
//
//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/PFClusterSpecificAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidate.h"

//____________________________________________________________________________||
reco::PFClusterMET PFClusterSpecificAlgo::addInfo(edm::Handle<edm::View<reco::Candidate> > PFClusterCandidates, CommonMETData met)
{  
  const LorentzVector p4(met.mex , met.mey, 0.0, met.met);
  const Point vtx(0.0,0.0,0.0);
  reco::PFClusterMET pfClusterMET(met.sumet, p4, vtx);
  return pfClusterMET;
}
