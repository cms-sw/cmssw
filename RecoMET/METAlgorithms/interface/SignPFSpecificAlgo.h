// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignPFSpecificAlgo
//
/**\class SignPFSpecificAlgo

 Description: Organizes information specific to the Jet-based significance for
              Particle Flow MET

*/
//
// Authors: A. Khukhunaishvili (Cornell), L. Gibbons (Cornell)
// First Implementation: November 11, 2011
//
//
#ifndef METAlgorithms_SignPFSpecificAlgo_h
#define METAlgorithms_SignPFSpecificAlgo_h

//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "TMatrixD.h"

//____________________________________________________________________________||
namespace metsig
{

  class SignPFSpecificAlgo
  {
    
  public:
    SignPFSpecificAlgo();
    ~SignPFSpecificAlgo() { }

    void setResolutions( metsig::SignAlgoResolutions *resolutions);
    void addPFJets(edm::Handle<edm::View<reco::PFJet> > PFJets);
    void addPFCandidate(reco::PFCandidatePtr pf);
    void useOriginalPtrs(const edm::ProductID& productID);
    TMatrixD getSignifMatrix() const {return algo_.getSignifMatrix();}

  private:
    metsig::SignAlgoResolutions *resolutions_;
    std::set<reco::CandidatePtr> clusteredParticlePtrs_;
    metsig::significanceAlgo algo_;
  };

}

//____________________________________________________________________________||
#endif // METAlgorithms_SignPFSpecificAlgo_h
