#ifndef SiStripMonitorClient_SiStripCertificationInfo_h
#define SiStripMonitorClient_SiStripCertificationInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripCertificationInfo
// 
/**\class SiStripCertificationInfo SiStripCertificationInfo.h DQM/SiStripMonitorCluster/interface/SiStripCertificationInfo.h

 Description: 
      Checks the # of SiStrip FEDs from DAQ
 Usage:
    <usage>

*/
//
//          Author:  Suchandra Dutta
//         reated:  Mon Feb 16 19200:00 CET 2009
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
class SiStripDetCabling;

class SiStripCertificationInfo: public DQMEDHarvester {

 public:

  /// Constructor
  SiStripCertificationInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripCertificationInfo();

 private:

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// End Of Luminosity
  void dqmEndLuminosityBlock(DQMStore::IBooker & , DQMStore::IGetter & , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  
private:

  void bookSiStripCertificationMEs(DQMStore::IBooker & , DQMStore::IGetter &);
  void resetSiStripCertificationMEs(DQMStore::IBooker & , DQMStore::IGetter &);
  void fillSiStripCertificationMEs(DQMStore::IBooker & , DQMStore::IGetter &);

  void fillDummySiStripCertification(DQMStore::IBooker & , DQMStore::IGetter &);

  void fillSiStripCertificationMEsAtLumi(DQMStore::IBooker & , DQMStore::IGetter &);

  struct SubDetMEs{
    MonitorElement* det_fractionME;
    std::string folder_name;
    std::string subdet_tag;
    int n_layer;
  };

  MonitorElement * SiStripCertification;
  MonitorElement * SiStripCertificationMap; 
  std::map<std::string, SubDetMEs> SubDetMEsMap;
  MonitorElement * SiStripCertificationSummaryMap;

  bool sistripCertificationBooked_;
  unsigned long long m_cacheID_;

  edm::ESHandle< SiStripDetCabling > detCabling_;

  int nFEDConnected_;

  const TrackerTopology* tTopo;

  };
#endif
