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
#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Configuration.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"

namespace cscdqm {

  /**
   * @class EventProcessor
   * @brief Object used to process Events and compute statistics
   */
  class EventProcessor {

// ===================================================================================================
// General stuff 
// ===================================================================================================

    public:
      
      EventProcessor(Configuration* const p_config);

      /**
       * @brief  Destructor
       */
      ~EventProcessor() { }

      void init();
      void updateFractionHistos();
      void updateEfficiencyHistos();

    private:
      
      void processExaminer(const uint16_t *data, const uint32_t dataSize, bool& eventDenied); 
      void processDDU(const CSCDDUEventData& data);
      void processCSC(const CSCEventData& data, const int dduID);

      void calcEMUFractionHisto(const HistoId& result, const HistoId& set, const HistoId& subset);

      const bool getEMUHisto(const HistoId& histo, MonitorObject*& me);
      const bool getDDUHisto(const HistoId& histo, const HwId& dduID, MonitorObject*& me);
      const bool getCSCHisto(const HistoId& histo, const HwId& crateID, const HwId& dmbSlot, MonitorObject*& me);
      const bool getCSCHisto(const HistoId& histo, const HwId& crateID, const HwId& dmbSlot, const HwId& adId, MonitorObject*& me);
      const bool getParHisto(const HistoId& histo, MonitorObject*& me);

      const bool getCSCFromMap(const unsigned int& crateId, const unsigned int& dmbId, unsigned int& cscType, unsigned int& cscPosition) const;

      /** Pointer to Global Configuration */
      Configuration* config;

      /** Detector efficiency manipulation object */
      Summary summary;

      /** CSC DCC Examiner object */
      CSCDCCExaminer binChecker;

      std::map<uint32_t, uint32_t> L1ANumbers;
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

#endif      

// ===================================================================================================
// Global ONLY stuff 
// ===================================================================================================

#ifdef DQMGLOBAL

    private:

      bool bCSCEventCounted;

    public:

      void processEvent(const edm::Event& e, const edm::InputTag& inputTag);

#endif      

  };

}

#endif
