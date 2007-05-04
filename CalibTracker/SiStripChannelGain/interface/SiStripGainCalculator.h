#ifndef CalibTracker_SiStripChannelGain_SiStripGainCalculator_h
#define CalibTracker_SiStripChannelGain_SiStripGainCalculator_h
// -*- C++ -*-
//
// Package:    SiStripGainCalculator
// Class:      SiStripGainCalculator
// 
/**\class SiStripGainCalculator SiStripGainCalculator.cc CalibTracker/SiStripChannelGain/src/SiStripGainCalculator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiStripGainCalculator.h,v 1.1 2007/05/02 12:46:46 gbruno Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "CondCore/DBCommon/interface/Time.h"

class SiStripApvGain;

class SiStripGainCalculator : public edm::EDAnalyzer {

public:
  explicit SiStripGainCalculator(const edm::ParameterSet&);
  ~SiStripGainCalculator();


private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void beginRun(const edm::Run &, const edm::EventSetup &);
  virtual void endRun(const edm::Run &, const edm::EventSetup &);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&)=0;
  virtual void endJob() ;

protected:

  virtual  SiStripApvGain * gainCalibrationPointer(); //pointer to be set to actual object by the derived concrete calculators 

private:
  
  //  edm::RunNumber_t runNumber_;  // Current run number. not used.
  bool SinceAppendMode_; // till or since append mode 
  cond::Time_t Time_; //time until which the DB object is valid. It is taken from the time of the first event analyzed. The end of the validity is infinity. However as soon as a new DB object with a later start time is inserted, the end time of this one becomes the start time of the new one. 

  SiStripApvGain * SiStripApvGain_;

};

#endif
