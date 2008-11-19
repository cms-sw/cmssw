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
#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoProviderIf.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"

#define TAG_EMU "EMU"
#define TAG_DDU "DDU_%d"
#define TAG_CSC "CSC_%03d_%02d"

namespace cscdqm {

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

  typedef enum BinCheckerCRCType { 
    ALCT, 
    CFEB, 
    TMB 
  };

  typedef std::map<std::string, uint32_t> CSCCounters;

  class EventProcessor {

// ===================================================================================================
// General stuff 
// ===================================================================================================

    public:
      
      EventProcessor(HistoProvider* p_histoProvider);
      ~EventProcessor();

      void blockHisto(const HistoType histo);

      void setDDUCheckMask(const uint32_t mask) { dduCheckMask = mask; }
      const uint32_t getDDUCheckMask() const { return dduCheckMask; }
      void setBinCheckMask(const uint32_t mask) { binCheckMask = mask; }
      uint32_t getBinCheckMask() const { return binCheckMask; }

      void setBinCheckerCRC(const BinCheckerCRCType crc, const bool value);
      void setBinCheckerOutput(const bool value);

      void updateFractionHistos();
      void updateEfficiencyHistos(EffParametersType& effParams);

    private:
      
      void processExaminer(const uint16_t *data, const uint32_t dataSize, bool& eventDenied); 
      void processDDU(const CSCDDUEventData& data);
      void processCSC(const CSCEventData& data, const int dduID);

      void calcEMUFractionHisto(const HistoType result, const HistoType set, const HistoType subset);

      const bool histoNotBlocked(const HistoType histo) const;

      const bool getEMUHisto(const HistoType histo, MonitorObject* me, const bool ref = false);
      const bool getDDUHisto(const int dduID, const HistoType histo, MonitorObject* me, const bool ref = false);
      const bool getCSCHisto(const int crateID, const int dmbSlot, const HistoType histo, MonitorObject* me, const int adId = 0, const bool ref = false);

      std::set<HistoType> blocked;
      HistoProvider* histoProvider;
      Summary summary;

      uint32_t nEvents; 
      uint32_t nBadEvents; 
      uint32_t nGoodEvents; 
      uint32_t nCSCEvents;
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

      void processEvent(const edm::Event& e);
      void setInputTag(const edm::InputTag p_inputTag) { inputTag = p_inputTag; }

    private:

      edm::InputTag inputTag;

#endif      

  };

}

#endif
