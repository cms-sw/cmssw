#ifndef L1GCTBASICHISTOGRAMMER_H
#define L1GCTBASICHISTOGRAMMER_H

/** \class L1GctBasicHistogrammer
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

class L1GctBasicHistogrammer : public L1GctHistogrammer {

 public:

  ///constructor
  L1GctBasicHistogrammer(TFile* tf=0, const std::string dir="default");

  ///destructor
  virtual ~L1GctBasicHistogrammer();

  ///event processor
  virtual void fillHistograms(const GctOutputData gct);

 private:

  TH1F allJetsRank;
  TH1F allJetsEta;
  TH1F allJetsPhi;
  TH1F allJetsGctEta;
  TH1F metValue;
  TH1F metAngle;

};

#endif
