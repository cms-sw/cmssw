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
#include <typeinfo>
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
static const char PARAM_BOOKING_FILE[]   = "BookingFile";

static const char INPUT_TAG_LABEL[]      = "source";

static const char DIR_SUMMARY[]          = "CSC/Summary/";
static const char DIR_DDU[]              = "CSC/DDU/";
static const char DIR_CSC[]              = "CSC/Chamber/";
static const char DIR_EVENTINFO[]        = "CSC/EventInfo/";
static const char DIR_SUMMARY_CONTENTS[] = "CSC/EventInfo/reportSummaryContents/";

/// Local Types

static const std::type_info& EMUHistoT   = typeid(cscdqm::EMUHistoType);
static const std::type_info& DDUHistoT   = typeid(cscdqm::DDUHistoType);
static const std::type_info& CSCHistoT   = typeid(cscdqm::CSCHistoType);
static const std::type_info& ParHistoT   = typeid(cscdqm::ParHistoType);

typedef std::map<std::string, CSCMonitorObject*> MOCacheMap;
typedef std::set<std::string>                    bookedHistoSet;
typedef std::vector<cscdqm::CSCHistoType>        bookedCSCSet;
typedef std::bitset<32>                          Bitset32;


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
    cscdqm::Dispatcher       *dispatcher;
    DQMStore                 *dbe;
    edm::InputTag             inputTag;
    MOCacheMap                moCache;
    bookedHistoSet            bookedHisto;
    bookedCSCSet              bookedCSCs;

    /** Pointer to crate mapping from database **/
    const CSCCrateMap* pcrate;

    /** Fractional histograms update stuff */
    Bitset32        fractUpdateKey;
    uint32_t        fractUpdateEvF;
    
  ///
  // MonitorObjectProvider Implementation
  ///

  public:

    const bool getHisto(const cscdqm::HistoType& histo, cscdqm::MonitorObject*& mo);

    void getCSCFromMap(const unsigned int crateId, const unsigned int dmbId, unsigned int& cscType, unsigned int& cscPosition) const;
    const CSCDetId getCSCDetId(const unsigned int crateId, const unsigned int dmbId) const { 
      return pcrate->detId(crateId, dmbId, 0, 0); 
    }
    const bool nextCSC(unsigned int& iter, unsigned int& crateId, unsigned int& dmbId) const;

    cscdqm::MonitorObject* bookInt (const std::string &name);
    cscdqm::MonitorObject *bookInt (const std::string &name, const int default_value);
    cscdqm::MonitorObject* bookFloat (const std::string &name);
    cscdqm::MonitorObject *bookFloat (const std::string &name, const float default_value);
    cscdqm::MonitorObject* bookString (const std::string &name, const std::string &value);
    cscdqm::MonitorObject* book1D (const std::string &name, const std::string &title, int nchX, double lowX, double highX);
    cscdqm::MonitorObject* book2D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY);
    cscdqm::MonitorObject* book3D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ);
    cscdqm::MonitorObject* bookProfile (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, const char *option = "s");
    cscdqm::MonitorObject* bookProfile2D (const std::string &name, const std::string &title, int nchX, double lowX, double highX, int nchY, double lowY, double highY, int nchZ, double lowZ, double highZ, const char *option = "s");

    void afterBook (cscdqm::MonitorObject*& me);

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
