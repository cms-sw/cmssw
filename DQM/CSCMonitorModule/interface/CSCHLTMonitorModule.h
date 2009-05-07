/*
 * =====================================================================================
 *
 *       Filename:  CSCHLTMonitorModule.h
 *
 *    Description: Main object of CSC HLT DQM Monitor 
 *
 *        Version:  1.0
 *        Created:  09/15/2008 01:30:00 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCHLTMonitorModule_H
#define CSCHLTMonitorModule_H

/**
 * Include Section
 */

#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <bitset>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DQM/CSCMonitorModule/interface/CSCUtility.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"

/**
 * Macro Section
 */

#define LOGERROR(cat)      edm::LogError (cat)
#define LOGWARNING(cat)    edm::LogWarning (cat)
#define LOGINFO(cat)       edm::LogInfo (cat)
#define LOGDEBUG(cat)      LogDebug (cat)

#define FED_FOLDER "FEDIntegrity"

/**
 * Type Definition Section
 */

typedef std::bitset<32> Bitset32;
typedef std::map<const std::string, MonitorElement*> MeMap;

/**
 * @class CSCHLTMonitorModule
 * @brief HLT Level CSC DQM module
 */
class CSCHLTMonitorModule: public edm::EDAnalyzer {

  public:

    CSCHLTMonitorModule(const edm::ParameterSet& ps);
    virtual ~CSCHLTMonitorModule();

  protected:

    void beginJob(const edm::EventSetup& c);
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void setup();
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
    void endRun(const edm::Run& r, const edm::EventSetup& c);
    void endJob();

  private:

    /** Histogram filling and calculation methods */
    void monitorEvent(const edm::Event& e) ;

    /** Monitor Elements */
    MeMap mes;

    /** Global Module-wide parameters  */
    edm::ParameterSet  parameters;
    DQMStore*          dbe;
    std::string        monitorName;
    std::string        rootDir;

    /** If histos have been initialized? **/
    bool init;

    /** Source related stuff */
    edm::InputTag   inputObjectsTag;

    /** Examiner and its stuff */
    unsigned int    examinerMask;
    bool            examinerForce;
    bool            examinerOutput;
    Bitset32        examinerCRCKey;

    /** FED mapping, increments, etc. */
    uint32_t        nEvents;
    uint32_t        fedIdMin;
    uint32_t        fedIdMax;
    
};

#endif
