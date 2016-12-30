#ifndef SiStripMonitorClient_SiStripDcsInfo_h
#define SiStripMonitorClient_SiStripDcsInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripDcsInfo
// 
/**\class SiStripDcsInfo SiStripDcsInfo.h DQM/SiStripMonitorCluster/interface/SiStripDcsInfo.h

 Description: 
      Checks the # of SiStrip FEDs from DAQ
 Usage:
    <usage>

*/
//
//          Author:  Suchandra Dutta
//         Created:  Mon Feb 16 19:00:00 CET 2009
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
class SiStripDetVOff;
class SiStripDetCabling; 

class SiStripDcsInfo: public DQMEDHarvester {

 public:

  /// Constructor
  SiStripDcsInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripDcsInfo();

 private:

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Begin Luminosity Block
  void dqmBeginLuminosityBlock(DQMStore::IBooker & , DQMStore::IGetter & , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) ;

  /// End Of Luminosity
  void dqmEndLuminosityBlock(DQMStore::IBooker & , DQMStore::IGetter & , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  void bookStatus(DQMStore::IBooker & , DQMStore::IGetter &);
  void readStatus(edm::EventSetup const&);
  void readCabling(edm::EventSetup const&);
  void addBadModules(DQMStore::IBooker & , DQMStore::IGetter &);
  void fillStatus(DQMStore::IBooker & , DQMStore::IGetter &);
  void fillDummyStatus(DQMStore::IBooker & , DQMStore::IGetter &);

  MonitorElement * DcsFraction_;

  struct SubDetMEs{
    std::string folder_name;
    MonitorElement* DcsFractionME;
    int TotalDetectors;
    std::vector<uint32_t> FaultyDetectors;
  };

  std::map <std::string, SubDetMEs> SubDetMEsMap;
  unsigned long long m_cacheIDCabling_;
  unsigned long long m_cacheIDDcs_;
  bool bookedStatus_;

  edm::ESHandle<SiStripDetVOff> siStripDetVOff_;
  int  nFEDConnected_;
  
  const TrackerTopology* tTopo;

  edm::ESHandle< SiStripDetCabling > detCabling_;
};
#endif
