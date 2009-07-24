
#ifndef _LAST0PRODUCERDQM_H
#define _LAST0PRODUCERDQM_H

// -*- C++ -*-
//
// Package:    Alignment/LaserAlignment
// Class:      LaserAlignmentT0ProducerDQM
// 

//
// DQM module for the
// Laser Alignment AlCaReco producer
// (LaserAlignmentT0Producer from Alignment/LaserAlignment)
//

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "Alignment/LaserAlignment/src/LASGlobalData.cc" // template
#include "Alignment/LaserAlignment/src/LASGlobalLoop.h"



class LaserAlignmentT0ProducerDQM : public edm::EDAnalyzer {

 public:

  explicit LaserAlignmentT0ProducerDQM( const edm::ParameterSet& );
  ~LaserAlignmentT0ProducerDQM();
  virtual void beginJob( edm::EventSetup const& );
  virtual void endJob( void );
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
 private:

  void FillFromRawDigis( const edm::DetSetVector<SiStripRawDigi>& );
  void FillFromProcessedDigis( const edm::DetSetVector<SiStripDigi>& );
  void FillDetectorId( void );

  DQMStore* theDqmStore;
  edm::ParameterSet theConfiguration;
  std::vector<edm::ParameterSet> theDigiProducerList;

  std::vector<int> tecDoubleHitDetId;
  LASGlobalData<int> detectorId;

  unsigned int theLowerAdcThreshold;
  unsigned int theUpperAdcThreshold;

  //2D
  MonitorElement* nSignalsAT;
  MonitorElement* nSignalsTECPlusR4;
  MonitorElement* nSignalsTECPlusR6;
  MonitorElement* nSignalsTECMinusR4;
  MonitorElement* nSignalsTECMinusR6;

};

#endif
