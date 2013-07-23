#ifndef EcalLocalRecoTask_H
#define EcalLocalRecoTask_H

/*
 * \file EcalLocalRecoTask.h
 *
 * $Id: EcalLocalRecoTask.h,v 1.4 2010/07/21 04:23:25 wmtan Exp $
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
#include <string>

class EcalLocalRecoTask: public edm::EDAnalyzer
{


 public:
  
  /// Constructor
  EcalLocalRecoTask(const edm::ParameterSet& ps);
  
  /// Destructor
  ~EcalLocalRecoTask();
  
 protected:
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  // BeginJob
  void beginJob();
  
  // EndJob
  void endJob(void);
  
 private:
  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;  

  bool verbose_;
  
  DQMStore* dbe_;
  
  std::string outputFile_;

  std::string recHitProducer_;
  std::string ESrecHitProducer_;
  
  std::string EBrechitCollection_;
  std::string EErechitCollection_;
  std::string ESrechitCollection_;

  std::string uncalibrecHitProducer_;
  std::string EBuncalibrechitCollection_;
  std::string EEuncalibrechitCollection_;

  std::string digiProducer_;
  std::string EBdigiCollection_;
  std::string EEdigiCollection_;
  std::string ESdigiCollection_;

  MonitorElement* meEBUncalibRecHitMaxSampleRatio_;
  MonitorElement* meEBUncalibRecHitPedestal_;
  MonitorElement* meEBUncalibRecHitOccupancy_;
  MonitorElement* meEBRecHitSimHitRatio_;

};

#endif
