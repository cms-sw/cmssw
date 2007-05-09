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
// $Id: SiStripGainCalculator.h,v 1.2 2007/05/04 20:22:34 gbruno Exp $
//
//


// system include files
#include <memory>
#include <string>

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

  void beginJob(const edm::EventSetup&);
  virtual void algoBeginJob(const edm::EventSetup&){};
  void beginRun(const edm::Run &, const edm::EventSetup &);
  virtual void algoBeginRun(const edm::Run &, const edm::EventSetup &){};
  void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
  virtual void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};

  void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void algoAnalyze(const edm::Event&, const edm::EventSetup&){};


  void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
  virtual void algoEndLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){};

  void endRun(const edm::Run &, const edm::EventSetup &);
  virtual void algoEndRun(const edm::Run &, const edm::EventSetup &){};

  void endJob() ;
  virtual void algoEndJob(){};


  void storeOnDb(SiStripApvGain *);
  void setTime();

  // to be implemented by concrete algorithm. Must return an object created with "new". The algo looses control on it (must not "delete" it!) 
  virtual SiStripApvGain * getNewObject()=0;
  //  virtual void resetOldData()=0;

protected:

  void storeOnDbNow(); // to be called by the concrete algorithm only if it support the algodriven mode; it will trigger a call of getnewObject, but only if algoDrivenMode is chosen

  cond::Time_t timeOfLastIOV(){return Time_;}



private:
  
  bool SinceAppendMode_; // till or since append mode 

  bool LumiBlockMode_; //LumiBlock since/till time
  bool RunMode_; //
  bool JobMode_;
  bool AlgoDrivenMode_;

  std::string Record_;
  cond::Time_t Time_; //time until which the DB object is valid. It is taken from the time of the first event analyzed. The end of the validity is infinity. However as soon as a new DB object with a later start time is inserted, the end time of this one becomes the start time of the new one. 

  bool setSinceTime_;


};

#endif
