
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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASGlobalLoop.h"

class LaserAlignmentT0ProducerDQM : public DQMEDAnalyzer {

 public:

  explicit LaserAlignmentT0ProducerDQM( const edm::ParameterSet& );
  ~LaserAlignmentT0ProducerDQM();

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze( const edm::Event&, const edm::EventSetup&) override;

 private:

  void FillFromRawDigis( const edm::DetSetVector<SiStripRawDigi>& );
  void FillFromProcessedDigis( const edm::DetSetVector<SiStripDigi>& );
  void FillDetectorId( void );

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
