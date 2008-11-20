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

#ifdef DQMLOCAL


  void EventProcessor::processEvent(const char* data, const int32_t dataSize, const uint32_t errorStat, const int32_t nodeNumber) {

    nEvents++;
    nCSCEvents++;

    MonitorObject* me = 0;
    if (getEMUHisto(EMU_ALL_READOUT_ERRORS, me)) {
      if(errorStat != 0) {
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
    if (((uint32_t) errorStat & dduCheckMask) > 0) {
      eventDenied = true;
    }

    const uint16_t *tmp = reinterpret_cast<const uint16_t *>(data);
    
    processExaminer(tmp, dataSize / sizeof(short), eventDenied);

    if (!eventDenied) {
      nGoodEvents++;
      CSCDDUEventData dduData((short unsigned int*) tmp, &binChecker);
      processDDU(dduData);
    }

  }

#endif

#ifdef DQMGLOBAL


  void EventProcessor::processEvent(const edm::Event& e, const edm::InputTag& inputTag) {

    nEvents++;
    bCSCEventCounted = false;

    // get a handle to the FED data collection
    // actualy the FED_EVENT_LABEL part of the event
    edm::Handle<FEDRawDataCollection> rawdata;
    e.getByLabel(inputTag, rawdata);

    // run through the DCC's 
    for (int id = FEDNumbering::getCSCFEDIds().first; id <= FEDNumbering::getCSCFEDIds().second; ++id) {

      // Take a reference to this FED's data and
      // construct the DCC data object
      const FEDRawData& fedData = rawdata->FEDData(id);
  
      //if fed has data then unpack it
      if (fedData.size() >= 32) {

        if (!bCSCEventCounted) {
          nCSCEvents++;
          bCSCEventCounted = true;
        }

        const uint16_t *data = (uint16_t *) fedData.data();
        bool eventDenied = false;
        
        processExaminer(data, long(fedData.size() / 2), eventDenied);

        if (!eventDenied) {
          nGoodEvents++;
          CSCDCCEventData dccData((short unsigned int*) data);
          const std::vector<CSCDDUEventData> & dduData = dccData.dduData();
          for (int ddu = 0; ddu < (int)dduData.size(); ddu++) {
            processDDU(dduData[ddu]);
          }
        }

      }

    }

  }

#endif

}
