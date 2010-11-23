/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor_processEvent.cc
 *
 *    Description:  EventProcessor Object Event entry methods
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:47:11 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"

namespace cscdqm {

  /**
   * @brief  Common Local and Global DQM function to be called before processing Event
   */
  void EventProcessor::preProcessEvent() {

    config->incNEvents();
    EmuEventDisplayWasReset = false;

  }

#ifdef DQMLOCAL


  /**
   * @brief  Process event (Local DQM)
   * @param  data Event Data buffer
   * @param  dataSize Event Data buffer size
   * @param  errorStat Error status received by reading DAQ buffer
   * @param  nodeNumber DAQ node number
   * @return 
   */
  void EventProcessor::processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber) {

    preProcessEvent();

    config->incNEventsCSC();

    MonitorObject* me = 0;
    if (getEMUHisto(h::EMU_ALL_READOUT_ERRORS, me)) {
      if (errorStat != 0) {
        me->Fill(nodeNumber, 1);
        for (unsigned int i = 0; i < 16; i++) {
          if ((errorStat >> i) & 0x1) {
            me->Fill(nodeNumber, i + 2);
          }
        }
      } else {
        me->Fill(nodeNumber, 0);
      }
    }

    bool eventDenied = false;
    if (((uint32_t) errorStat & config->getDDU_CHECK_MASK()) > 0) {
      eventDenied = true;
    }

    const uint16_t *tmp = reinterpret_cast<const uint16_t *>(data);
    const uint16_t  tmpSize = dataSize / sizeof(short);
    
    /** CSC DCC Examiner object */
    CSCDCCExaminer binChecker;
    binChecker.crcALCT(config->getBINCHECKER_CRC_ALCT());
    binChecker.crcTMB(config->getBINCHECKER_CRC_CLCT());
    binChecker.crcCFEB(config->getBINCHECKER_CRC_CFEB());
    binChecker.modeDDU(config->getBINCHECKER_MODE_DDU());

    if (config->getBINCHECKER_OUTPUT()) {
      binChecker.output1().show();
      binChecker.output2().show();
    } else {
      binChecker.output1().hide();
      binChecker.output2().hide();
    }

    binChecker.setMask(config->getBINCHECK_MASK());
    
    if (binChecker.check(tmp, tmpSize) < 0) {

      /** No ddu trailer found - force checker to summarize errors by adding artificial trailer */
      const uint16_t dduTrailer[4] = { 0x8000, 0x8000, 0xFFFF, 0x8000 };
      const uint16_t *tmp = dduTrailer;
      binChecker.check(tmp, uint32_t(4));

    }

    CSCDCCFormatStatusDigi formatStatusDigi(nodeNumber, config->getBINCHECK_MASK(),
      binChecker.getMask(), 
      binChecker.errors(),
      binChecker.errorsDetailedDDU(),
      binChecker.errorsDetailed(),
      binChecker.payloadDetailed(),
      binChecker.statusDetailed());

    if (processExaminer(binChecker, formatStatusDigi)) {

      config->incNEventsGood();

      if (config->getPROCESS_DDU()) {
        CSCDDUEventData dduData((short unsigned int*) tmp, &binChecker);
        processDDU(dduData, binChecker);
      }

    }

  }

#endif

#ifdef DQMGLOBAL


  /**
   * @brief  Process event (Global DQM)
   * @param  e Event object
   * @param  inputTag Tag to search Event Data in
   * @param  standby HW standby status indicator
   * @return 
   */
  void EventProcessor::processEvent(const edm::Event& e, const edm::InputTag& inputTag) {

    preProcessEvent();

    bCSCEventCounted = false;

    edm::Handle<FEDRawDataCollection> rawdata;
    if (!e.getByLabel(inputTag, rawdata)) {
      LOG_WARN << "No product: " << inputTag << " in stream";
      return;
    }

    /*
    const edm::InputTag formatStatusCollectionTag("MuonCSCDCCFormatStatusDigi");
    bool processFormatStatusDigi = true;
    edm::Handle<CSCDCCFormatStatusDigiCollection> formatStatusColl;
    if (!e.getByLabel(formatStatusCollectionTag, formatStatusColl)) {
      LOG_WARN << "No product: " << formatStatusCollectionTag << " in stream";
      processFormatStatusDigi = false;
    }
    */

    // run through the DCC's 
    for (int id = FEDNumbering::MINCSCFEDID; id <= FEDNumbering::MAXCSCFEDID; ++id) {

      // Take a reference to this FED's data and
      // construct the DCC data object
      const FEDRawData& fedData = rawdata->FEDData(id);
  
      //if fed has data then unpack it
      if (fedData.size() >= 32) {

        // Count in CSC Event!
        if (!bCSCEventCounted) {
          config->incNEventsCSC();
          bCSCEventCounted = true;
        }

        // Filling in FED Entries histogram
        MonitorObject* mo = 0;
        if (getEMUHisto(h::EMU_FED_ENTRIES, mo)) mo->Fill(id); 

        const uint16_t *data = (uint16_t *) fedData.data();
        const size_t dataSize = fedData.size() / 2;
        const short unsigned int* udata = (short unsigned int*) fedData.data();
        
        /** CSC DCC Examiner object */
        CSCDCCExaminer binChecker;
        binChecker.crcALCT(config->getBINCHECKER_CRC_ALCT());
        binChecker.crcTMB(config->getBINCHECKER_CRC_CLCT());
        binChecker.crcCFEB(config->getBINCHECKER_CRC_CFEB());
        binChecker.modeDDU(config->getBINCHECKER_MODE_DDU());

        if (config->getBINCHECKER_OUTPUT()) {
          binChecker.output1().show();
          binChecker.output2().show();
        } else {
          binChecker.output1().hide();
          binChecker.output2().hide();
        }

        binChecker.setMask(config->getBINCHECK_MASK());

        if (binChecker.check(data, dataSize) < 0) {

          if (getEMUHisto(h::EMU_FED_FATAL, mo)) mo->Fill(id);

        } else {

          CSCDCCFormatStatusDigi formatStatusDigi(id, config->getBINCHECK_MASK(),
              binChecker.getMask(), 
              binChecker.errors(),
              binChecker.errorsDetailedDDU(),
              binChecker.errorsDetailed(),
              binChecker.payloadDetailed(),
              binChecker.statusDetailed());

          if (processExaminer(binChecker, formatStatusDigi)) {

            config->incNEventsGood();

            if (binChecker.warnings() != 0) {
              if (getEMUHisto(h::EMU_FED_NONFATAL, mo)) mo->Fill(id);
            } 

            if (config->getPROCESS_DDU()) {

              CSCDCCEventData dccData(const_cast<short unsigned int*>(udata), &binChecker);
              const std::vector<CSCDDUEventData> & dduData = dccData.dduData();
              
              for (int ddu = 0; ddu < (int)dduData.size(); ddu++) {
                processDDU(dduData[ddu], binChecker);
              }
            }

          } else {

            if (getEMUHisto(h::EMU_FED_FORMAT_FATAL, mo)) mo->Fill(id);

          }

        }

      }

    }

  }

#endif

#undef ECHO_FUNCTION

}
