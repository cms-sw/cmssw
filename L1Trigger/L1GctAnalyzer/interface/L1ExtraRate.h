// -*- C++ -*-
//
// Package:    L1GctAnalyzer
// Class:      L1ExtraRate
// 
/**\class L1ExtraRate L1ExtraRate.cc L1Trigger/L1GctAnalyzer/src/L1ExtraRate.cc

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

#ifndef L1EXTRARATE_H
#define L1EXTRARATE_H

#include "DataFormats/Candidate/interface/LeafCandidate.h"


class L1ExtraRate {

 public:

  L1ExtraRate(string name);
  ~L1ExtraRate();

  /// set weight (for use with eg binned QCD datasets)
  void setWeight(double w);

  /// fill histograms
  void fill(reco::LeafCandidate& l1cand);

 private:

  // rate as fn of threshold
  TH1F rate_;

}
