// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      METAlgo
//
/**\class METAlgo METAlgo.h RecoMET/METAlgorithms/interface/METAlgo.h

 Description: Calculates MET for given input

 Implementation:
     [Notes on implementation]
*/
//
// Original Authors:  Michael Schmitt, Richard Cavanaugh The University of Florida
//          Created:  May 14, 2005
//
//

//____________________________________________________________________________||
#ifndef METAlgo_h
#define METAlgo_h

//____________________________________________________________________________||
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

//____________________________________________________________________________||
class METAlgo {
public:
  METAlgo() {}
  virtual ~METAlgo() {}
  CommonMETData run(const edm::View<reco::Candidate>& candidates,
                    double globalThreshold = 0.0,
                    edm::ValueMap<float> const* weights = nullptr);
};

//____________________________________________________________________________||
#endif  // METAlgo_h
