
// -*- C++ -*-
//
// Package:    L1GctAnalyzer
// Class:      L1ExtraDistributions
// 
/**\class L1ExtraDistributions L1ExtraDistributions.cc L1Trigger/L1GctAnalyzer/src/L1ExtraDistributions.cc

 Description: Makes histograms from comparison between L1Extra candidate and MC particle

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  23 Nov 2006
// $Id:
//
//

#ifndef L1EXTRADISTRIBUTIONS_H
#define L1EXTRADISTRIBUTIONS_H

#include "DataFormats/Candidate/interface/LeafCandidate.h"

class L1ExtraDistributions {

 public:

  // use name as base for histograms
  L1ExtraDistributions(string name);

  ~L1ExtraDistributions();

  // fill the histos
  void fill(reco::LeafCandidate& l1cand);

 private:

  TH1F etDist_;
  TH1F etaDist_;
  TH1F phiDist_;

}
