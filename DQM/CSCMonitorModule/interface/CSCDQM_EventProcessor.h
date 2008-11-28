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

#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"

namespace cscdqm {

  /**
   * @brief  Structure to provide a set of efficiency parameters.
   */
  typedef struct EffParametersType {
    double cold_threshold;
    double cold_sigfail;
    double hot_threshold;
    double hot_sigfail;
    double err_threshold;
    double err_sigfail;
    double nodata_threshold;
    double nodata_sigfail;
  };

  /**
   * @brief  Switch on/off CRC check for various levels
   */
  typedef enum BinCheckerCRCType { 
    ALCT, 
    CFEB, 
    TMB 
  };

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
      
      EventProcessor(MonitorObjectProvider* p_provider);
      ~EventProcessor();

      void setDDUCheckMask(const uint32_t mask) { dduCheckMask = mask; }
      const uint32_t getDDUCheckMask() const { return dduCheckMask; }
      void setBinCheckMask(const uint32_t mask) { binCheckMask = mask; }
      uint32_t getBinCheckMask() const { return binCheckMask; }

      void setBinCheckerCRC(const BinCheckerCRCType crc, const bool value);
      void setBinCheckerOutput(const bool value);

      void updateFractionHistos();
      void updateEfficiencyHistos(EffParametersType& effParams);

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
      uint32_t dduCheckMask;
      uint32_t binCheckMask;
      uint32_t dduBinCheckMask; 

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
