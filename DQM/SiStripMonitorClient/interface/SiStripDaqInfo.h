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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class MonitorElement;
class SiStripFedCabling;
class TrackerTopology;

class SiStripDaqInfo: public DQMEDHarvester {

 public:

  /// Constructor
  SiStripDaqInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripDaqInfo();

 private:

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// End Of Luminosity
  void dqmEndLuminosityBlock(DQMStore::IBooker & , DQMStore::IGetter & , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;


private:
  void readFedIds(const edm::ESHandle<SiStripFedCabling>& fedcabling, edm::EventSetup const& iSetup);
  void readSubdetFedFractions(DQMStore::IBooker & , DQMStore::IGetter & , std::vector<int>& fed_ids);
  void bookStatus(DQMStore::IBooker & , DQMStore::IGetter &);
  void fillDummyStatus(DQMStore::IBooker & , DQMStore::IGetter &);
  void findExcludedModule(DQMStore::IBooker & , DQMStore::IGetter & , unsigned short fed_id, const TrackerTopology* tTopo
);

  std::map<std::string,std::vector<unsigned short> > subDetFedMap;

  MonitorElement * DaqFraction_;

  struct SubDetMEs{
    MonitorElement* DaqFractionME;
    int ConnectedFeds;
  };

  std::map <std::string, SubDetMEs> SubDetMEsMap;

  unsigned long long m_cacheID_;
  int nFedTotal;
  int nFEDConnected;
  bool bookedStatus_;
  std::vector<int> FedsInIds;

  edm::ESHandle< SiStripFedCabling > fedCabling_;
  const TrackerTopology* tTopo;

};
#endif
