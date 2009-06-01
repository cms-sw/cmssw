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
  bCSCEventCounted = false;
  if(nEvents %1000 == 0) {
    LOGINFO("monitorEvent") << " # of events = " << nEvents << ", # of CSC events = " << nCSCEvents << std::endl;
  }

  // Get a handle to the FED data collection
  // actualy the FED_EVENT_LABEL part of the event
  edm::Handle<FEDRawDataCollection> rawdata;
  if (!e.getByLabel( inputObjectsTag, rawdata)) {
    // LOGWARNING("e.getByLabel") << "No product: " << inputObjectsTag << " in FEDRawDataCollection";
    return; 
  }

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

    // LOGWARNING("MonitorEvent") << "Event No." << nEvents << " size = " << fedData.size() << std::endl;

    //if fed has data then unpack it
    if ( fedData.size() >= 32 ) {

      // Fed contains valid CSC data - lets count this in
      if (!bCSCEventCounted) {
        nCSCEvents++;
        bCSCEventCounted = true;
      }
       
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

     // LOGWARNING("MonitorEvent") << "Event No." << nEvents << " is " << std::boolalpha << goodEvent << std::endl;

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

  bool goodEvent = true;

  if ((examiner.errors() & examinerMask) > 0) {
    goodEvent = false;
  }

  std::map<int,long> payloads = examiner.payloadDetailed();
  for(std::map<int,long>::const_iterator chamber = payloads.begin(); chamber != payloads.end(); chamber++) {

    //int ChamberID = chamber->first;
    int CrateID = (chamber->first>>4) & 0xFF;
    int DMBSlot = chamber->first & 0xF;

    if (CrateID == 255) { continue; }

    if (MEEMU("DMB_Reporting", me)) me->Fill(CrateID, DMBSlot);

    int CSCtype   = 0;
    int CSCposition = 0;
    if (!getCSCFromMap(CrateID, DMBSlot, CSCtype, CSCposition )) continue;

    if (CSCtype && CSCposition && MEEMU("CSC_Reporting", me)) me->Fill(CSCposition, CSCtype);

    long payload = chamber->second;
    int cfeb_dav = (payload>>7) & 0x1F;
    int alct_dav = (payload>>5) & 0x1;
    int tmb_dav = (payload>>6) & 0x1; 
      
    if (alct_dav == 0) {
      if (CSCtype && CSCposition && MEEMU("CSC_wo_ALCT", me)) me->Fill(CSCposition, CSCtype);
      if (MEEMU("DMB_wo_ALCT", me)) me->Fill(CrateID, DMBSlot);
    }
     
    if (tmb_dav == 0) {
      if (CSCtype && CSCposition && MEEMU("CSC_wo_CLCT", me)) me->Fill(CSCposition, CSCtype);
      if (MEEMU("DMB_wo_CLCT", me)) me->Fill(CrateID, DMBSlot);
    }

    if (cfeb_dav == 0) {
      if (CSCtype && CSCposition && MEEMU("CSC_wo_CFEB", me)) me->Fill(CSCposition, CSCtype);
      if (MEEMU("DMB_wo_CFEB", me)) me->Fill(CrateID,DMBSlot);
    }
      
  }

  if ((examiner.errors() != 0) || (examiner.warnings() != 0)) {

    std::map<int,long> checkerErrors = examiner.errorsDetailed();
    for( std::map<int,long>::const_iterator chamber = checkerErrors.begin(); chamber != checkerErrors.end() ; chamber++ ){
    
      //int ChamberID = chamber->first;
      int CrateID = (chamber->first>>4) & 0xFF;
      int DMBSlot = chamber->first & 0xF;

      if ((CrateID == 255) || (chamber->second & 0x80)) continue; // = Skip chamber detection if DMB header is missing (Error code 6)

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
      if (!getCSCFromMap(CrateID, DMBSlot, CSCtype, CSCposition)) continue;

      if (isCSCError && CSCtype && CSCposition && MEEMU("CSC_Format_Errors", me)) {
        me->Fill(CSCposition, CSCtype);
      }

      if (goodEvent && isCSCError && CSCtype && CSCposition && MEEMU("CSC_Unpacked_with_errors", me)) {
        me->Fill(CSCposition, CSCtype);
      }

    }

  }
  
  std::map<int,long> statuses = examiner.statusDetailed();
  for(std::map<int,long>::const_iterator chamber = statuses.begin(); chamber != statuses.end(); chamber++) {

    //int ChamberID = chamber->first;
    int CrateID = (chamber->first>>4) & 0xFF;
    int DMBSlot = chamber->first & 0xF;
    std::string cscTag(Form("CSC_%03d_%02d", CrateID, DMBSlot));
    if (CrateID == 255) {continue;}
    
    int CSCtype   = 0;
    int CSCposition = 0;
    if (!getCSCFromMap(CrateID, DMBSlot, CSCtype, CSCposition )) continue;

    int anyInputFull = chamber->second & 0x3F;
    if(anyInputFull){
      if (CSCtype && CSCposition && MEEMU("CSC_DMB_input_fifo_full", me)) me->Fill(CSCposition, CSCtype);
      if (MEEMU("DMB_input_fifo_full", me)) me->Fill(CrateID, DMBSlot);
    }

    int anyInputTO = (chamber->second >> 7) & 0x3FFF;
    if(anyInputTO){
      if (CSCtype && CSCposition && MEEMU("CSC_DMB_input_timeout", me)) me->Fill(CSCposition, CSCtype);
      if (MEEMU("DMB_input_timeout", me)) me->Fill(CrateID, DMBSlot);
    }

    if (chamber->second & (1<<22)) {
      if (MEEMU("DMB_Format_Warnings", me)) me->Fill(CrateID, DMBSlot);
      if (CSCtype && CSCposition && MEEMU("CSC_Format_Warnings", me)) me->Fill(CSCposition, CSCtype);
    }

  }

  return goodEvent;
}
