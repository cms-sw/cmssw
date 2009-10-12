#ifndef SiStripMonitorClient_SiStripDaqInfo_h
#define SiStripMonitorClient_SiStripDaqInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripDaqInfo
// 
/**\class SiStripDaqInfo SiStripDaqInfo.h DQM/SiStripMonitorCluster/interface/SiStripDaqInfo.h

 Description: 
      Checks the # of SiStrip FEDs from DAQ
 Usage:
    <usage>

*/
//
//          Author:  Suchandra Dutta
//         Created:  Thu Dec 11 17:50:00 CET 2008
//

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
class MonitorElement;
class SiStripFedCabling;

class SiStripDaqInfo: public edm::EDAnalyzer {

 public:

  /// Constructor
  SiStripDaqInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripDaqInfo();

 private:

  /// BeginJob
  void beginJob(edm::EventSetup const& eSetup);

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Begin Of Luminosity
                                                                               
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);

  /// Analyze

  void analyze(edm::Event const&, edm::EventSetup const&);



private:
  void readFedIds(const edm::ESHandle<SiStripFedCabling>& fedcabling);
  void readSubdetFedFractions(std::vector<int>& fed_ids);
  void bookStatus();
  void fillDummyStatus();

  std::map<std::string,std::vector<unsigned short> > subDetFedMap;

  DQMStore* dqmStore_;
  MonitorElement * DaqFraction_;

  struct SubDetMEs{
    MonitorElement* DaqFractionME;
    int TotalFed;
    int ConnectedFeds;
  };

  std::map <std::string, SubDetMEs> SubDetMEsMap;

  unsigned long long m_cacheID_;
  int nFedTotal;
  bool bookedStatus_;
};
#endif
