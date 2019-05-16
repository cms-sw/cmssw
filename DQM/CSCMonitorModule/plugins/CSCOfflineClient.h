/*
 * =====================================================================================
 *
 *       Filename:  CSCOfflineClient.h
 *
 *    Description:  CSC Offline module that preocess merged histograms and
 *    creates/updates fractional and efficiency objects.
 *
 *        Version:  1.0
 *        Created:  09/20/2009 01:36:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */


#ifndef CSCOfflineClient_H
#define CSCOfflineClient_H

/// Global stuff
#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <set>

/// DQM Framework stuff
// #include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>

/// CSC Framework stuff
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"

/// CSCDQM Framework stuff
#include "CSCDQM_Logger.h"
#include "CSCDQM_Configuration.h"
#include "CSCDQM_Dispatcher.h"
#include "CSCMonitorModule.h"

/// Local stuff
#include "CSCMonitorObject.h"

/// Local Constants
//static const char DIR_EVENTINFO[]        = "CSC/EventInfo/";
//static const char DIR_DCSINFO[]          = "CSC/EventInfo/DCSContents/";
//static const char DIR_DAQINFO[]          = "CSC/EventInfo/DAQContents/";
//static const char DIR_CRTINFO[]          = "CSC/EventInfo/CertificationContents/";

/**
 * @class CSCOfflineClient
 * @brief CSC Offline DQM Client that uses CSCDQM Framework 
 */
class CSCOfflineClient: public DQMEDHarvester, public cscdqm::MonitorObjectProvider {
 
  /**
   * Global stuff
   */

  public:

    CSCOfflineClient(const edm::ParameterSet& ps);
    ~CSCOfflineClient() override;

  private:

    cscdqm::Configuration     config;
    cscdqm::Dispatcher       *dispatcher;
    // DQMStore                 *dbe;
    DQMStore::IBooker        *ibooker;
    std::vector<std::string> maskedHW;

  /**
   * MonitorObjectProvider Implementation
   */

  public:

    bool getCSCDetId(const unsigned int crateId, const unsigned int dmbId, CSCDetId& detId) const override{ return false; }
    cscdqm::MonitorObject *bookMonitorObject (const cscdqm::HistoBookRequest& p_req) override;

  /** 
   * EDAnalyzer Implementation
   */ 

  protected:

    // void beginJob() { }
    // void beginRun(const edm::Run& r, const edm::EventSetup& c) { }
    void setup() { }
    //void analyze(const edm::Event& e, const edm::EventSetup& c) { }
    // void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) { } 
    // void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) { }
    // void endRun(const edm::Run& r, const edm::EventSetup& c);
    // void endJob() { }
    void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

};

#endif
