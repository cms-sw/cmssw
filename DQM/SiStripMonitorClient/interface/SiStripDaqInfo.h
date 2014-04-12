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
class TrackerTopology;

class SiStripDaqInfo: public edm::EDAnalyzer {

 public:

  /// Constructor
  SiStripDaqInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripDaqInfo();

 private:

  /// BeginJob
  void beginJob();

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// End Of Luminosity
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Analyze
  void analyze(edm::Event const&, edm::EventSetup const&);


private:
  void readFedIds(const edm::ESHandle<SiStripFedCabling>& fedcabling, edm::EventSetup const& iSetup);
  void readSubdetFedFractions(std::vector<int>& fed_ids, edm::EventSetup const& iSetup);
  void bookStatus();
  void fillDummyStatus();
  void findExcludedModule(unsigned short fed_id, const TrackerTopology* tTopo
);

  std::map<std::string,std::vector<unsigned short> > subDetFedMap;

  DQMStore* dqmStore_;
  MonitorElement * DaqFraction_;

  struct SubDetMEs{
    MonitorElement* DaqFractionME;
    int ConnectedFeds;
  };

  std::map <std::string, SubDetMEs> SubDetMEsMap;

  unsigned long long m_cacheID_;
  int nFedTotal;
  bool bookedStatus_;

  edm::ESHandle< SiStripFedCabling > fedCabling_;
};
#endif
