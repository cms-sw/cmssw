/*
 * =====================================================================================
 *
 *       Filename:  CSCHLTMonitorModule_monitorEvent.cc
 *
 *    Description:  HLT Monitor Event method implementation. This method is the primary
 *    entry point for Event data.
 *
 *        Version:  1.0
 *        Created:  09/15/2008 02:17:37 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCHLTMonitorModule.h"

/**
 * @brief  Monitoring function that receives Events 
 * @param  e Event
 * @param  c EventSetup
 * @return
 */
void CSCHLTMonitorModule::monitorEvent(const edm::Event& e){

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

    if (examinerCRCKey.test(0)) examiner.crcALCT(1);
    if (examinerCRCKey.test(1)) examiner.crcCFEB(1);
    if (examinerCRCKey.test(2)) examiner.crcTMB(1);

    if (examinerOutput) {
      examiner.output1().show();
      examiner.output2().show();
    } else {
      examiner.output1().hide();
      examiner.output2().hide();
    }

    // Take a reference to this FED's data and
    // construct the DCC data object
    const FEDRawData& fedData = rawdata->FEDData(id);

    //if fed has data then unpack it
    if ( fedData.size() >= 32 ) {

      unsigned int index = 0;
      if (fedIndex(id, index)) mes["FEDEntries"]->Fill(index); 

      const short unsigned int *data = (short unsigned int *) fedData.data();

      if (examiner.check(data, long(fedData.size()/2)) < 0 ) {
        const uint16_t dduTrailer[4] = { 0x8000, 0x8000, 0xFFFF, 0x8000 };
        data = dduTrailer;
        examiner.check(data, uint32_t(4));
      }

      if ((examiner.errors() & examinerMask) > 0) {
        if (fedIndex(id, index)) mes["FEDFatal"]->Fill(index); 
      }

      if (examiner.warnings() != 0) {
        if (fedIndex(id, index)) mes["FEDNonFatal"]->Fill(index); 
      }
     
    }
  }

}

