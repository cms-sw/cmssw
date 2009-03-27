/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModuleCmn.h
 *
 *    Description:  Updated CSC Monitor module
 *
 *        Version:  1.0
 *        Created:  11/13/2008 01:36:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */


#ifndef CSCMonitorModuleCmn_H
#define CSCMonitorModuleCmn_H

/// Global stuff
#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <set>

/// DQM Framework stuff
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/EventSetup.h>

/// CSC Framework stuff
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"

/// CSCDQM Framework stuff
#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Dispatcher.h"

/// Local stuff
#include "DQM/CSCMonitorModule/interface/CSCMonitorObject.h"

/// Local Constants
static const char INPUT_TAG_LABEL[]      = "source";
static const char DIR_EVENTINFO[]        = "CSC/EventInfo/";

/**
 * @class CSCMonitorModuleCmn
 * @brief Common CSC DQM Module that uses CSCDQM Framework  
 */
class CSCMonitorModuleCmn: public edm::EDAnalyzer, public cscdqm::MonitorObjectProvider {
 
  ///
  // Global stuff
  ///

  public:

    CSCMonitorModuleCmn(const edm::ParameterSet& ps);
    virtual ~CSCMonitorModuleCmn();

  private:

    cscdqm::Configuration     config;
    cscdqm::Dispatcher        dispatcher;
    DQMStore                 *dbe;
    edm::InputTag             inputTag;

    /** Pointer to crate mapping from database **/
    const CSCCrateMap* pcrate;

  ///
  // MonitorObjectProvider Implementation
  ///

  public:

    const CSCDetId getCSCDetId(const unsigned int crateId, const unsigned int dmbId) const { 
      return pcrate->detId(crateId, dmbId, 0, 0); 
    }
    cscdqm::MonitorObject *bookMonitorObject (const cscdqm::HistoBookRequest& p_req); 

  /// 
  // EDAnalyzer Implementation
  /// 

  protected:

    void beginJob(const edm::EventSetup& c);
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void setup();
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
    void endRun(const edm::Run& r, const edm::EventSetup& c);
    void endJob();

};

#endif
