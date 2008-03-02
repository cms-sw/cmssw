#ifndef EcalLocalRecoTask_H
#define EcalLocalRecoTask_H

/*
 * \file EcalLocalRecoTask.h
 *
 * $Id: EcalLocalRecoTask.h,v 1.1 2006/04/07 12:38:49 meridian Exp $
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace cms;
using namespace edm;
using namespace std;

class EcalLocalRecoTask: public EDAnalyzer
{


 public:
  
  /// Constructor
  EcalLocalRecoTask(const ParameterSet& ps);
  
  /// Destructor
  ~EcalLocalRecoTask();
  
 protected:
  
  /// Analyze
  void analyze(const Event& e, const EventSetup& c);
  
  // BeginJob
  void beginJob(const EventSetup& c);
  
  // EndJob
  void endJob(void);
  
 private:
  typedef map<uint32_t,float,less<uint32_t> >  MapType;  

  bool verbose_;
  
  DQMStore* dbe_;
  
  string outputFile_;

  string recHitProducer_;
  string ESrecHitProducer_;
  
  string EBrechitCollection_;
  string EErechitCollection_;
  string ESrechitCollection_;

  string uncalibrecHitProducer_;
  string EBuncalibrechitCollection_;
  string EEuncalibrechitCollection_;

  string digiProducer_;
  string EBdigiCollection_;
  string EEdigiCollection_;
  string ESdigiCollection_;

  MonitorElement* meEBUncalibRecHitMaxSampleRatio_;
  MonitorElement* meEBUncalibRecHitPedestal_;
  MonitorElement* meEBUncalibRecHitOccupancy_;
  MonitorElement* meEBRecHitSimHitRatio_;

};

#endif
