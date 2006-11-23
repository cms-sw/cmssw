// -*- C++ -*-
//
// Package:    L1GctAnalyzer
// Class:      L1ExtraComparator
// 
/**\class L1ExtraComparator L1ExtraComparator.cc L1Trigger/L1GctAnalyzer/src/L1ExtraComparator.cc

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

#ifndef L1EXTRACOMPARATOR_H
#define L1EXTRACOMPARATOR_H

#include "DataFormats/Candidate/interface/LeafCandidate.h"


class L1ExtraComparator {

 public:

  // name is base for histograms
  L1ExtraComparator(string name);

  ~L1ExtraComparator();

  // fill the histos
  void fill(reco::LeafCandidate& l1cand, Particle& mc);

 private:

  // Et resolution
  TH1F etDiff_;  // fill with Et(L1)-Et(MC)
  TH1F scaledEtDiff_;  // fill with Et(L1)-Et(MC) / Et(MC)

  // position resolution
  TH1F etaDiff_; // eta(L1)-eta(MC)
  TH1F phiDiff_; // phi(L1)-phi(MC)
  TH1F rDiff_; // difference in R between L1 and MC

};

#endif

