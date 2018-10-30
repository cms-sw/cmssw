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
//
//


#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include <vector>
#include <memory>

class SiStripGainRandomCalculator : public ConditionDBWriter<SiStripApvGain> {

public:

  explicit SiStripGainRandomCalculator(const edm::ParameterSet&);
  ~SiStripGainRandomCalculator() override;

private:

  void algoAnalyze(const edm::Event &, const edm::EventSetup &) override;

  std::unique_ptr<SiStripApvGain> getNewObject() override;

private:

  double meanGain_;
  double sigmaGain_;
  double minimumPosValue_;

  std::vector< std::pair<uint32_t, unsigned short> > detid_apvs_;
  unsigned long long m_cacheID_;
  bool printdebug_;

};
#endif
