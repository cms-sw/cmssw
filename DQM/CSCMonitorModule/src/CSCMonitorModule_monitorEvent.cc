/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule_monitorEvent.cc
 *
 *    Description:  Monitor Event method implementation. This method is the primary
 *    entry point for Event data.
 *
 *        Version:  1.0
 *        Created:  04/18/2008 04:17:37 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"

/**
 * @brief  Monitoring function that receives Events 
 * @param  e Event
 * @param  c EventSetup
 * @return
 */
void CSCMonitorModule::monitorEvent(const edm::Event& e){

  nEvents++;
  if(nEvents %1000 == 0) {
    LOGINFO("monitorEvent") << " # of events = " << nEvents << std::endl;
  }

  // Get a handle to the FED data collection
  // actualy the FED_EVENT_LABEL part of the event
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel( inputObjectsTag, rawdata);

  // Lets run through the DCC's 
  for (int id = FEDNumbering::getCSCFEDIds().first; id <= FEDNumbering::getCSCFEDIds().second; ++id) {

    // Implement and set examiner
    CSCDCCExaminer examiner;

    if(examinerCRCKey.test(0)) examiner.crcALCT(1);
    if(examinerCRCKey.test(1)) examiner.crcCFEB(1);
    if(examinerCRCKey.test(2)) examiner.crcTMB(1);

    if(examinerOutput) {
      examiner.output1().show();
      examiner.output2().show();
    } else {
      examiner.output1().hide();
      examiner.output2().hide();
    }

    // Take a reference to this FED's data and
    // construct the DCC data object
    const FEDRawData& fedData = rawdata->FEDData(id);

    LOGDEBUG("MonitorEvent") << "Event No." << nEvents << " size = " << fedData.size() << std::endl;

    //if fed has data then unpack it
    if ( fedData.size() >= 32 ) {
       
/*  
  
  Removed as of not useful anymore (VR, 2008-04-29)

      // Filling in Buffer size histo with slight correction ;)
      if (MEEMU("Buffer_Size", me)) me->Fill(fedData.size() - 32);

*/ 

      const short unsigned int *data = (short unsigned int *) fedData.data();

      // If Event has not yet been passed via Examiner - lets do it now.
      // Examiner will set up the flag if event is good or not.
      if (examiner.check(data, long(fedData.size()/2)) < 0 ) {
        // No ddu trailer found - force checker to summarize errors by adding artificial trailer
        const uint16_t dduTrailer[4] = { 0x8000, 0x8000, 0xFFFF, 0x8000 };
        data = dduTrailer;
        examiner.check(data, uint32_t(4));
      }

      bool goodEvent = monitorExaminer(examiner);
     
      // If event is OK then proceed with other procedures...
      if (goodEvent) {
        CSCDCCEventData dccData((short unsigned int *) fedData.data());
        monitorDCC(dccData);
      } 

     LOGDEBUG("MonitorEvent") << "Event No." << nEvents << " is " << std::boolalpha << goodEvent << std::endl;

    }
  }

}


/**
 * @brief  Examiner object analyzer
 * @param  examiner Examiner object to monitor
 * @return true if event is good following examiner data, false otherwise
 */
bool CSCMonitorModule::monitorExaminer(CSCDCCExaminer& examiner) {

  MonitorElement* me = NULL;

  if (MEEMU("All_DDUs_Format_Errors", me)) {
    std::vector<int> DDUs = examiner.listOfDDUs();
    for (std::vector<int>::iterator ddu_itr = DDUs.begin(); ddu_itr != DDUs.end(); ++ddu_itr) {
      if (*ddu_itr != 0xFFF) {
        long errs = examiner.errorsForDDU(*ddu_itr);
        int dduID = (*ddu_itr)&0xFF;
        std::string dduTag = Form("DDU_%d", dduID);
        if (errs != 0) {
          for(int i=0; i<examiner.nERRORS; i++) { // run over all errors
            if ((errs>>i) & 0x1 ) me->Fill(dduID, i+1);
          }
        } else {
          me->Fill(dduID, 0);
        }
      }
    }
  }

/*  
  
  Removed as of not useful anymore (VR, 2008-04-29)

  if (examiner.errors() != 0) {
    if (MEEMU("BinaryChecker_Errors", me)) {
      for(int i=0; i < examiner.nERRORS; i++) { // run over all errors
        if( examiner.error(i) ) me->Fill(0., i);
      }
    }
  }

  if(examiner.warnings() != 0) {
    if (MEEMU("BinaryChecker_Warnings", me)) {
      for(int i=0; i<examiner.nWARNINGS; i++) { // run over all warnings
        if( examiner.warning(i) ) me->Fill(0., i);
      }
    }
  }

*/

  bool goodEvent = true;

  if ((examiner.errors() & examinerMask) > 0) {
    goodEvent = false;
  }

  if ((examiner.errors() != 0) || (examiner.warnings() != 0)) {

    std::map<int,long> checkerErrors = examiner.errorsDetailed();
    for( std::map<int,long>::const_iterator chamber = checkerErrors.begin(); chamber != checkerErrors.end() ; chamber++ ){
    
      //int ChamberID = chamber->first;
      int CrateID = (chamber->first>>4) & 0xFF;
      int DMBSlot = chamber->first & 0xF;

      if ((CrateID == 255) || (chamber->second & 0x80)) continue; // = Skip chamber detection if DMB header is missing (Error code 6)
      if (CrateID > 60 || DMBSlot > 10) continue;

      bool isCSCError = false;
      for(int bit=5; bit<24; bit++) {
        if( chamber->second & (1<<bit) ) {
          isCSCError = true;
        }
        if( chamber->second & (1<<25) ) {
          isCSCError = true;
        }
      }

      if (isCSCError && MEEMU("DMB_Format_Errors", me)) {
        me->Fill(CrateID, DMBSlot);
      }

      if (goodEvent && isCSCError && MEEMU("DMB_Unpacked_with_errors", me)) {
        me->Fill(CrateID, DMBSlot);
      }

      int CSCtype   = 0;
      int CSCposition = 0;
      getCSCFromMap(CrateID, DMBSlot, CSCtype, CSCposition);

      if (isCSCError && CSCtype && CSCposition && MEEMU("CSC_Format_Errors", me)) {
        me->Fill(CSCposition, CSCtype);
      }

      if (goodEvent && isCSCError && CSCtype && CSCposition && MEEMU("CSC_Unpacked_with_errors", me)) {
        me->Fill(CSCposition, CSCtype);
      }

    }

    std::map<int,long> checkerWarnings  = examiner.warningsDetailed();
    for( std::map<int,long>::const_iterator chamber = checkerWarnings.begin(); chamber != checkerWarnings.end() ; chamber++ ){

      //int ChamberID = chamber->first;
      int CrateID = (chamber->first>>4) & 0xFF;
      int DMBSlot = chamber->first & 0xF;

      if (CrateID ==255) continue;

      bool isCSCWarning = false;
      for(int bit=1; bit<2; bit++) {
        if( chamber->second & (1<<bit) ) {
          isCSCWarning = true;
        }
      }

      if (isCSCWarning && MEEMU("DMB_Format_Warnings", me)) {
        me->Fill(CrateID, DMBSlot);
      }

      int CSCtype   = 0;
      int CSCposition = 0;
      getCSCFromMap(CrateID, DMBSlot, CSCtype, CSCposition );
      if (isCSCWarning && CSCtype && CSCposition && MEEMU("CSC_Format_Warnings", me)) {
        me->Fill(CSCposition, CSCtype);
      }

      if (goodEvent && isCSCWarning && CSCtype && CSCposition && MEEMU("CSC_Unpacked_with_warnings", me)) {
        me->Fill(CSCposition, CSCtype);
      }
    }
  }

  return goodEvent;
}
