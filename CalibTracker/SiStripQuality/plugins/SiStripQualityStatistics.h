#ifndef SiStripQualityStatistics_H
#define SiStripQualityStatistics_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h" 

#include <sstream>

class SiStripQualityStatistics : public edm::EDAnalyzer {

 public:
  explicit SiStripQualityStatistics( const edm::ParameterSet& );
  ~SiStripQualityStatistics(){};
  
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();
 
 private:

  void SetBadComponents(int,int,SiStripQuality::BadComponent&);

  unsigned long long m_cacheID_;
  std::string dataLabel_;
  std::string TkMapFileName_;
  edm::FileInPath fp_;
  bool saveTkHistoMap_;
  unsigned long long runNumberA_;
  //Global Info
  int NTkBadComponent[4]; //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  int NBadComponent[4][19][4];  
  //legend: NBadComponent[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
  //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
  //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  std::stringstream ssV[4][19];

  TrackerMap * tkMap, *tkMapFullIOVs;
  SiStripDetInfoFileReader* reader;
  TkHistoMap* tkhisto;
};
#endif
