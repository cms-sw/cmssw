/*
class: PFSpecificAlgo.cc
description:  MET made from Particle Flow candidates
authors: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
date: 10/27/08
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace reco;
using namespace std;

//--------------------------------------------------------------------------------------
// This algorithm adds Particle Flow specific global event information to the MET object
//--------------------------------------------------------------------------------------

reco::PFMET PFSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > PFCandidates, CommonMETData met)
{
  // Instantiate the container to hold the PF specific information
  SpecificPFMETData specific;
  // Initialize the container
  specific.NeutralEMFraction = 0.0;
  specific.NeutralHadFraction = 0.0;
  specific.ChargedEMFraction = 0.0;
  specific.ChargedHadFraction = 0.0;
  specific.MuonFraction = 0.0;


  if(!PFCandidates->size()) // if no Particle Flow candidates in the event
   {
     const LorentzVector p4( met.mex, met.mey, 0.0, met.met);
     const Point vtx(0.0, 0.0, 0.0 );
     PFMET specificPFMET( specific, met.sumet, p4, vtx);
     return specificPFMET;
   } 
  
  //Insert code to retreive / calculate specific pf data here:  




  const LorentzVector p4(met.mex , met.mey, 0.0, met.met);
  const Point vtx(0.0,0.0,0.0);
  PFMET specificPFMET( specific, met.sumet, p4, vtx );
  return specificPFMET;
}
