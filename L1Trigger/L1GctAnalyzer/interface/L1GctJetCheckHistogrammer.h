#ifndef L1GCTJETCHECKHISTOGRAMMER_H
#define L1GCTJETCHECKHISTOGRAMMER_H

/** \class L1GctJetCheckHistogrammer
 *
 * Books and fills histograms to check GCT
 * and calo trigger performance
 *
 * \author Greg Heath
 *
 * \date August 2006
 *
 */

#include "L1Trigger/L1GctAnalyzer/interface/L1GctHistogrammer.h"

#include "TH1.h"
#include "TH2.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

class L1GctJetCheckHistogrammer : public L1GctCorrelator<reco::GenJetCollection> {

 public:

  ///constructor
  L1GctJetCheckHistogrammer(TFile* tf, const std::string dir="default");

  ///destructor
  virtual ~L1GctJetCheckHistogrammer();

  ///event processor
  virtual void fillHistograms(const GctOutputData gct);

 private:

  static const int      NJETET,   NJETETA,   NJETPHI;
  static const double MINJETET, MINJETETA, MINJETPHI;
  static const double MAXJETET, MAXJETETA, MAXJETPHI;

  TH2F topJetRankVsGenEt;
  TH2F topJetEtaVsGen;
  TH2F topJetPhiVsGen;

};



#endif
