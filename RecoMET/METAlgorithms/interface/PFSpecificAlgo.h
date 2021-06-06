// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      PFSpecificAlgo
//
/**\class PFSpecificAlgo PFSpecificAlgo.h RecoMET/METAlgorithms/interface/PFSpecificAlgo.h

 Description: Adds Particle Flow specific information to MET

 Implementation:
     [Notes on implementation]
*/
//
// Original Authors:  R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
//          Created:  October 27, 2008
//
//
#ifndef METAlgorithms_PFMETInfo_h
#define METAlgorithms_PFMETInfo_h

//____________________________________________________________________________||
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/SpecificPFMETData.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//____________________________________________________________________________||
class PFSpecificAlgo {
public:
  PFSpecificAlgo() {}

  SpecificPFMETData run(const edm::View<reco::Candidate>& pfCands, edm::ValueMap<float> const* weights = nullptr);
};

//____________________________________________________________________________||
#endif  // METAlgorithms_PFMETInfo_h
