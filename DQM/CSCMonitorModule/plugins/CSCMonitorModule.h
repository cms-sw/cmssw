/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule.h
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


#ifndef CSCMonitorModule_H
#define CSCMonitorModule_H

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
#include "DataFormats/Scalers/interface/DcsStatus.h"

/// CSCDQM Framework stuff
#include "CSCDQM_Logger.h"
#include "CSCDQM_Configuration.h"
#include "CSCDQM_Dispatcher.h"

/// Local stuff
#include "CSCMonitorObject.h"

/// Local Constants
static const char INPUT_TAG_LABEL[]      = "source";
static const char DIR_EVENTINFO[]        = "CSC/EventInfo/";
static const char DIR_DCSINFO[]          = "CSC/EventInfo/DCSContents/";
static const char DIR_DAQINFO[]          = "CSC/EventInfo/DAQContents/";
static const char DIR_CRTINFO[]          = "CSC/EventInfo/CertificationContents/";

static const unsigned int MIN_CRATE_ID = 1;
static const unsigned int MAX_CRATE_ID = 60;
static const unsigned int MIN_DMB_SLOT = 1;
static const unsigned int MAX_DMB_SLOT = 10;

/**
 * @class CSCMonitorModule
 * @brief Common CSC DQM Module that uses CSCDQM Framework  
 */
class CSCMonitorModule: public edm::EDAnalyzer, public cscdqm::MonitorObjectProvider {
 
  /**
   * Global stuff
   */

  public:

    CSCMonitorModule(const edm::ParameterSet& ps);
    virtual ~CSCMonitorModule();

  private:

    cscdqm::Configuration     config;
    cscdqm::Dispatcher       *dispatcher;
    DQMStore                 *dbe;
    edm::InputTag             inputTag;
    bool                      prebookEffParams;
    bool                      processDcsScalers;

    /** Pointer to crate mapping from database **/
    const CSCCrateMap* pcrate;

  /**
   * MonitorObjectProvider Implementation
   */

  public:

    bool getCSCDetId(const unsigned int crateId, const unsigned int dmbId, CSCDetId& detId) const { 
      // Check parameter values
      if (crateId < MIN_CRATE_ID || crateId > MAX_CRATE_ID || dmbId < MIN_DMB_SLOT || dmbId > MAX_DMB_SLOT) {
        return false;
      }
      detId = pcrate->detId(crateId, dmbId, 0, 0);
      return (detId.rawId() != 0);
    }

    cscdqm::MonitorObject *bookMonitorObject (const cscdqm::HistoBookRequest& p_req); 

  /** 
   * EDAnalyzer Implementation
   */ 

  protected:

    void beginJob() { }
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void setup() { }
    void analyze(const edm::Event& e, const edm::EventSetup& c);
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) { }
    void endRun(const edm::Run& r, const edm::EventSetup& c) { }
    void endJob() { }

};

#endif
