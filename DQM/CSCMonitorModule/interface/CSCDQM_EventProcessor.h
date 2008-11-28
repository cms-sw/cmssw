/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor.h
 *
 *    Description:  Object which processes Event and provides Hits to
 *    HitHandler object.
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:26:04 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_EventProcessor_H
#define CSCDQM_EventProcessor_H

#include <set>
#include <string>
#include <math.h>

#include <TString.h>

#ifdef DQMGLOBAL

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"

#endif

#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Summary.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_StripClusterFinder.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectProvider.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"

namespace cscdqm {

  typedef std::map<std::string, uint32_t> CSCCounters;

  /**
   * @class EventProcessor
   * @brief Object used to process Events and compute statistics
   */
  class EventProcessor {

// ===================================================================================================
// General stuff 
// ===================================================================================================

    public:
      
      EventProcessor(const Configuration* p_config);
      ~EventProcessor() { }

      void updateFractionHistos();
      void updateEfficiencyHistos();

      const uint32_t getNEvents() const { return nEvents; } 
      const uint32_t getNCSCEvents() const { return nCSCEvents; }
      const uint32_t getNBadEvents() const { return nBadEvents; } 
      const uint32_t getNGoodEvents() const { return nGoodEvents; }

    private:
      
      void processExaminer(const uint16_t *data, const uint32_t dataSize, bool& eventDenied); 
      void processDDU(const CSCDDUEventData& data);
      void processCSC(const CSCEventData& data, const int dduID);

      void calcEMUFractionHisto(const HistoName& result, const HistoName& set, const HistoName& subset);

      const bool getEMUHisto(const HistoName& histo, MonitorObject*& me);
      const bool getDDUHisto(const int dduID, const HistoName& histo, MonitorObject*& me);
      const bool getCSCHisto(const int crateID, const int dmbSlot, const HistoName& histo, MonitorObject*& me);
      const bool getCSCHisto(const int crateID, const int dmbSlot, const HistoName& histo, MonitorObject*& me, const int adId);
      const bool getParHisto(const std::string& name, MonitorObject*& me);

      MonitorObjectProvider* provider;
      const Configuration*   config;
      Summary summary;

      uint32_t nEvents; 
      uint32_t nBadEvents; 
      uint32_t nGoodEvents; 
      uint32_t nCSCEvents;
      bool     bCSCEventCounted;

      uint32_t unpackedDMBcount;
      std::map<std::string, uint32_t> nDMBEvents;
      std::map<std::string, CSCCounters> cscCntrs;

      CSCDCCExaminer binChecker;

      std::map<uint32_t,uint32_t> L1ANumbers;
      uint32_t L1ANumber;
      uint32_t BXN;
      bool fFirstEvent;
      bool fCloseL1As; // Close L1A bit from DDU Trailer
      
// ===================================================================================================
// Local ONLY stuff 
// ===================================================================================================

#ifdef DQMLOCAL

    public:

      void processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber);

    private:

#endif      

// ===================================================================================================
// Global ONLY stuff 
// ===================================================================================================

#ifdef DQMGLOBAL

    public:

      void processEvent(const edm::Event& e, const edm::InputTag& inputTag);

    private:

#endif      

  };

}

#endif
