#ifndef CalibTracker_SiStripESProducers_SiStripNoiseDummyCalculator_h
#define CalibTracker_SiStripESProducers_SiStripNoiseDummyCalculator_h
// -*- C++ -*-
//
// Package:    SiStripPedestals
// Class:      SiStripNoiseDummyCalculator
// 
/**\class SiStripNoiseDummyCalculator SiStripNoiseDummyCalculator.cc CalibTracker/SiStripPedestals/src/SiStripNoiseDummyCalculator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giacomo Bruno
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripNoiseDummyCalculator.h,v 1.2 2008/03/04 15:59:22 giordano Exp $
//
//


#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include <vector>


class SiStripNoiseDummyCalculator : public ConditionDBWriter<SiStripNoises> {

public:

  explicit SiStripNoiseDummyCalculator(const edm::ParameterSet&);
  ~SiStripNoiseDummyCalculator();

private:

  void algoAnalyze(const edm::Event &, const edm::EventSetup &);

  SiStripNoises * getNewObject();

private:

  bool stripLengthMode_;
  
  //parameters for random noise generation. not used if Strip length mode is chosen
  double meanNoise_;
  double sigmaNoise_;
  double minimumPosValue_;

  //parameters for strip length proportional noise generation. not used if random mode is chosen
  double noiseStripLengthLinearSlope_;
  double noiseStripLengthLinearQuote_;
  double electronsPerADC_;


  std::map<uint32_t, std::pair<unsigned short, double> > detData_;
  unsigned long long m_cacheID_;

  bool printdebug_;


};
#endif
