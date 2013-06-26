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

#include "CSCDQM_Logger.h"
#include "CSCDQM_Summary.h"
#include "CSCDQM_StripClusterFinder.h"
#include "CSCDQM_Configuration.h"
#include "CSCDQM_Configuration.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"

#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"

namespace cscdqm {

  /**
   * Structure of standby flags
   */
  struct HWStandbyType {
  
    // if standby flag should be considered at all?
    // at the start it will be false, thus good for last value ;)
    bool process;

    // ME+
    bool MeP;

    // ME-
    bool MeM;

    HWStandbyType() {
      process = MeP = MeM = false;
    }

    void applyMeP(bool ready) {
      MeP = MeP || !ready;
    }

    void applyMeM(bool ready) {
      MeM = MeM || !ready;
    }

    bool fullStandby() const {
      return (MeM && MeP);
    }

    bool operator==(const HWStandbyType& t) const {
      return (t.MeP == MeP && t.MeM == MeM && t.process == process);
    }

    bool operator!=(const HWStandbyType& t) const {
      return !(*this == t);
    }

    const HWStandbyType& operator= (const HWStandbyType& t) {
      MeP = t.MeP;
      MeM = t.MeM;
      process = t.process;
      return *this;
    }

  };

  typedef std::map<CSCIdType, ExaminerStatusType> CSCExaminerMapType;
  typedef std::vector<DDUIdType>                  DDUExaminerVectorType;
  // typedef std::map<int, long> CSCExaminerMapType;
  // typedef std::vector<int>    DDUExaminerVectorType;

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
      void standbyEfficiencyHistos(HWStandbyType& standby);
      void writeShifterHistograms();

      unsigned int maskHWElements(std::vector<std::string>& tokens);

    private:
      
      bool processExaminer(const CSCDCCExaminer& binChecker); 
      bool processExaminer(const CSCDCCExaminer& binChecker, const CSCDCCFormatStatusDigi& digi);
      void processDDU(const CSCDDUEventData& data, const CSCDCCExaminer& binChecker);
      void processCSC(const CSCEventData& data, const int dduID, const CSCDCCExaminer& binChecker);

      void calcEMUFractionHisto(const HistoId& result, const HistoId& set, const HistoId& subset);

      const bool getEMUHisto(const HistoId& histo, MonitorObject*& me);
      const bool getFEDHisto(const HistoId& histo, const HwId& fedID, MonitorObject*& me);
      const bool getDDUHisto(const HistoId& histo, const HwId& dduID, MonitorObject*& me);
      const bool getCSCHisto(const HistoId& histo, const HwId& crateID, const HwId& dmbSlot, MonitorObject*& me);
      const bool getCSCHisto(const HistoId& histo, const HwId& crateID, const HwId& dmbSlot, const HwId& adId, MonitorObject*& me);
      const bool getParHisto(const HistoId& histo, MonitorObject*& me);
      void preProcessEvent();

      const bool getCSCFromMap(const unsigned int& crateId, const unsigned int& dmbId, unsigned int& cscType, unsigned int& cscPosition) const;
      void setEmuEventDisplayBit(MonitorObject*& mo, const unsigned int x, const unsigned int y, const unsigned int bit);
      void resetEmuEventDisplays();

      /** Pointer to Global Configuration */
      Configuration* config;

      /** Detector efficiency manipulation object */
      Summary summary;

      std::map<uint32_t, uint32_t> L1ANumbers;
      std::map<uint32_t, bool> fNotFirstEvent;
      uint32_t L1ANumber;
      uint32_t BXN;
      uint32_t cntDMBs; 	/// Total Number of DMBs per event from DDU Header DAV
      uint32_t cntCFEBs;	/// Total Number of CFEBs per event from DMB DAV 
      uint32_t cntALCTs;	/// Total Number of ALCTs per event from DMB DAV 
      uint32_t cntTMBs;		/// Total Number of TMBs per event from DMB DAV
      
	
      // bool fFirstEvent;
      bool fCloseL1As; // Close L1A bit from DDU Trailer
      bool EmuEventDisplayWasReset;
      
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
