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
// $Id: SiStripGainRandomCalculator.h,v 1.2 2007/05/04 20:22:34 gbruno Exp $
//
//


#include "CalibTracker/SiStripChannelGain/interface/SiStripGainCalculator.h"
#include <vector>

class SiStripGainRandomCalculator : public SiStripGainCalculator {

public:

  explicit SiStripGainRandomCalculator(const edm::ParameterSet&);
  ~SiStripGainRandomCalculator();


private:
  void algoBeginRun(const edm::Run &, const edm::EventSetup &);
  //  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&);
  //  virtual void endJob() ;

  SiStripApvGain * getNewObject();

private:


  double meanGain_;
  double sigmaGain_;
  double minimumPosValue_;
  std::vector< std::pair<uint32_t, unsigned short> > detid_apvs;
  bool printdebug_;

};
#endif
