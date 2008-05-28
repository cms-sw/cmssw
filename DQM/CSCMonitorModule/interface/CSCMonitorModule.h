/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule.h
 *
 *    Description: Main object of CSC DQM Monitor 
 *
 *        Version:  1.0
 *        Created:  04/18/2008 02:19:16 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCMonitorModule_H
#define CSCMonitorModule_H

/**
 * Include Section
 */

#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
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

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQM/CSCMonitorModule/interface/CSCSummary.h"

/**
 * Macro Section
 */

#define LOGERROR(cat)      edm::LogError (cat)
#define LOGWARNING(cat)    edm::LogWarning (cat)
#define LOGINFO(cat)       edm::LogInfo (cat)
#define LOGDEBUG(cat)      LogDebug (cat)

#define SUMMARY_FOLDER     "Summary/"
#define DDU_FOLDER         "DDUs/"
#define EVENTINFO_FOLDER   "EventInfo/"
#define SUMCONTENTS_FOLDER "EventInfo/reportSummaryContents/"

/**
 * Type Definition Section
 */

typedef std::map<std::string, std::string>     Histo;
typedef Histo::iterator                        HistoIter;
typedef std::map<std::string, Histo>           HistoDef;
typedef HistoDef::iterator                     HistoDefIter;
typedef std::map<std::string, HistoDef>        HistoDefMap;
typedef HistoDefMap::iterator                  HistoDefMapIter;

typedef std::bitset<36>                        BitsetDDU;
typedef std::bitset<32>                        Bitset32;

/**
 * Class Definition
 */

class CSCMonitorModule: public edm::EDAnalyzer {

  public:

    CSCMonitorModule(const edm::ParameterSet& ps);
    virtual ~CSCMonitorModule();

  protected:

    void beginJob(const edm::EventSetup& c);
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void setup();
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) ;
    void endRun(const edm::Run& r, const edm::EventSetup& c);
    void endJob();

  private:

    int loadCollection();    
    void printCollection();
    void book(const std::string prefix);
    const bool isMEValid(const std::string name, MonitorElement*& me);
    void getCSCFromMap(int crate, int slot, int& csctype, int& cscposition);

    /** Histogram filling and calculation methods */
    void monitorEvent(const edm::Event& e) ;
    void monitorDCC(const CSCDCCEventData& dccEvent);
    void monitorDDU(const CSCDDUEventData& dduEvent);
    void monitorCSC(const CSCEventData&    cscEvent, const int32_t& dduID);
    bool monitorExaminer(CSCDCCExaminer& examiner);
    void updateFracHistos();

    /** Global Module-wide parameters  */
    edm::ParameterSet  parameters;
    DQMStore*          dbe;
    std::string        monitorName;
    std::string        rootDir;
    std::string        bookingFile;

    /** If histos have been initialized? **/
    bool init;

    /** List of loaded (preloaded) DDU's */
    bool            hitBookDDU;

    /** Fractional histograms update stuff */
    Bitset32        fractUpdateKey;
    uint32_t        fractUpdateEvF;
    
    /** Source related stuff */
    edm::InputTag   inputObjectsTag;

    /** Examiner and its stuff */
    unsigned int    examinerMask;
    bool            examinerForce;
    bool            examinerOutput;
    Bitset32        examinerCRCKey;

    /** Pointer to crate mapping from database **/
    const CSCCrateMap* pcrate;

    /** Histogram collection */
    HistoDefMap collection;

    /** Histogram mapping, increments, etc. */
    uint32_t                     nEvents;
    std::map<std::string, int>   tmap;
    std::map<uint32_t,uint32_t>  L1ANumbers;

    /** Find histograms (aka previous macros) **/ 
    const bool MEEMU(const std::string name, MonitorElement*& me);
    const bool MEDDU(const unsigned int dduId, const std::string name, MonitorElement*& me);
    const bool MEEventInfo(const std::string name, MonitorElement*& me);
    const bool MEReportSummaryContents(const std::string name, MonitorElement*& me);

    /** CSC summary map */
    CSCSummary summary;
};

#endif
