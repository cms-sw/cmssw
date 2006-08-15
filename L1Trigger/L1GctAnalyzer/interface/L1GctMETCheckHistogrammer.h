#ifndef L1GCTMETCHECKHISTOGRAMMER_H
#define L1GCTMETCHECKHISTOGRAMMER_H

/** \class L1GctMETCheckHistogrammer
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

#include "DataFormats/METReco/interface/METCollection.h"

class L1GctMETCheckHistogrammer : public L1GctCorrelator<reco::METCollection> {

 public:

  ///constructor
  L1GctMETCheckHistogrammer(TFile* tf=0, const std::string dir="default");

  ///destructor
  virtual ~L1GctMETCheckHistogrammer();

  ///event processor
  virtual void fillHistograms(const GctOutputData gct);

 private:

  static const int      NGENMETVALUE,   NGENMETPHI;
  static const double MINGENMETVALUE, MINGENMETPHI;
  static const double MAXGENMETVALUE, MAXGENMETPHI;

  TH2F missingEtValueGCTVsGen;
  TH2F missingEtPhiGCTVsGen;
  TH2F missingEtRatioVsGCTJetEta;

};

#endif
