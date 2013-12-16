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
#ifndef METAlgo_h
#define METAlgo_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

class METAlgo 
{
public:
  METAlgo() {}
  virtual ~METAlgo() {}
  virtual CommonMETData run(const edm::View<reco::Candidate>& candidates, double globalThreshold = 0.0);
  virtual void run(const edm::View<reco::Candidate>& candidates, CommonMETData *met, double globalThreshold = 0.0);

  virtual CommonMETData run(edm::Handle<edm::View<reco::Candidate> > candidates, double globalThreshold = 0.0);
  virtual void run(edm::Handle<edm::View<reco::Candidate> > candidates, CommonMETData *met, double globalThreshold = 0.0);
};

#endif // METAlgo_h
