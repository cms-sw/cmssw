/*
class: PFClusterSpecificAlgo.cc
description:  MET made from Particle Flow candidates
authors: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
  date: 10/27/08
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/PFClusterSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidate.h"
using namespace reco;
using namespace std;

//--------------------------------------------------------------------------------------
// This algorithm adds Particle Flow specific global event information to the MET object
//--------------------------------------------------------------------------------------

reco::PFClusterMET PFClusterSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > PFClusterCandidates, CommonMETData met)
{  
  const LorentzVector p4(met.mex , met.mey, 0.0, met.met);
  const Point vtx(0.0,0.0,0.0);
  PFClusterMET specificPFClusterMET( met.sumet, p4, vtx );
  return specificPFClusterMET;
}
