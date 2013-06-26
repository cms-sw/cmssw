#ifndef CalibTracker_SiStripChannelGain_SiStripGainRandomCalculator_h
#define CalibTracker_SiStripChannelGain_SiStripGainRandomCalculator_h
// -*- C++ -*-
//
// Package:    SiStripApvGainCalculator
// Class:      SiStripApvGainCalculator
// 
/**\class SiStripApvGainCalculator SiStripApvGainCalculator.cc CalibTracker/SiStripChannelGain/src/SiStripApvGainCalculator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripGainRandomCalculator.h,v 1.1 2007/06/13 14:03:35 gbruno Exp $
//
//


#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>

class SiStripGainRandomCalculator : public ConditionDBWriter<SiStripApvGain> {

public:

  explicit SiStripGainRandomCalculator(const edm::ParameterSet&);
  ~SiStripGainRandomCalculator();

private:

  void algoAnalyze(const edm::Event &, const edm::EventSetup &);

  SiStripApvGain * getNewObject();

private:

  double meanGain_;
  double sigmaGain_;
  double minimumPosValue_;

  std::vector< std::pair<uint32_t, unsigned short> > detid_apvs_;
  unsigned long long m_cacheID_;
  bool printdebug_;

};
#endif
