#ifndef HGCalLocalRecoTask_H
#define HGCalLocalRecoTask_H

/*
 * \file HGCalLocalRecoTask.h
 *
 *
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


#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HGCalLocalRecoTask: public edm::EDAnalyzer
{


 public:
  
  /// Constructor
  HGCalLocalRecoTask(const edm::ParameterSet& ps);
  
  /// Destructor
  ~HGCalLocalRecoTask();
  
 protected:
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  // BeginJob
  void beginJob();
  
  // EndJob
  void endJob(void);
  
 private:
  //  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;  

  bool verbose_;

  DQMStore* dbe_;
  
  std::string outputFile_;

  std::string recHitProducer_;
  
  std::string HGCEErechitCollection_;
  std::string HGCHEFrechitCollection_;
  std::string HGCHEBrechitCollection_;

  std::string uncalibrecHitProducer_;
  std::string HGCEEuncalibrechitCollection_;
  std::string HGCHEFuncalibrechitCollection_;
  std::string HGCHEBuncalibrechitCollection_;

  std::string digiProducer_;
  std::string HGCEEdigiCollection_;
  std::string HGCHEFdigiCollection_;
  std::string HGCHEBdigiCollection_;

  MonitorElement* meHGCEEUncalibRecHitOccupancy_;
  MonitorElement* meHGCHEFUncalibRecHitOccupancy_;
  MonitorElement* meHGCHEBUncalibRecHitOccupancy_;

};

#endif
