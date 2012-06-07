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
// $Id: METProducer.h,v 1.29 2012/06/07 01:16:10 sakuma Exp $
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
  virtual void run(edm::Handle<edm::View<reco::Candidate> >, CommonMETData*,  double );
};

#endif // METAlgo_h
