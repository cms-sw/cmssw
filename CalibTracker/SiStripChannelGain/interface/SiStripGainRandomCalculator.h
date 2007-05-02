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
// $Id: SiStripApvGainCalculator.h,v 1.2 2007/02/20 16:11:16 dkcira Exp $
//
//


#include "CalibTracker/SiStripChannelGain/interface/SiStripGainCalculator.h"


class SiStripGainRandomCalculator : public SiStripGainCalculator {

public:

  explicit SiStripGainRandomCalculator(const edm::ParameterSet&);
  ~SiStripGainRandomCalculator();


private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&);
  //  virtual void endJob() ;

private:


  double meanGain_;
  double sigmaGain_;


};
