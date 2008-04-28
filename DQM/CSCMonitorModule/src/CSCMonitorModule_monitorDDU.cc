/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule_monitorDDU.cc
 *
 *    Description:  Monitor DDU method implementation.
 *
 *        Version:  1.0
 *        Created:  04/18/2008 03:39:16 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"
#include <math.h>

/**
 * @brief  MonitorDDU function that grabs DDUEventData and processes it.
 * @param  dduData DDU data
 * @return 
 */
void CSCMonitorModule::monitorDDU(const CSCDDUEventData& dduEvent){

  MonitorElement* me = NULL;
   
  // Get DDU header and trailer objects
  CSCDDUHeader dduHeader  = dduEvent.header();
  CSCDDUTrailer dduTrailer = dduEvent.trailer();

  int dduID = dduHeader.source_id()&0xFF; // Only 8bits are significant; format of DDU id is Dxx;

  if (MEEMU("All_DDUs_in_Readout", me)) me->Fill(dduID);

  int trl_word_count = dduTrailer.wordcount();
  if (trl_word_count > 0) {
    if (MEEMU("All_DDUs_Event_Size", me)) me->Fill(dduID, log10((double)trl_word_count));
  }

  if (MEEMU("All_DDUs_Average_Event_Size", me)) me->Fill(dduID, trl_word_count);

  std::map<uint32_t,uint32_t>::iterator it = L1ANumbers.find(dduID);
  if(it == L1ANumbers.end()) {
    L1ANumbers[dduID] = (int)(dduHeader.lvl1num());
  } else {
    int L1ANumber_previous_event = L1ANumbers[dduID];
    L1ANumbers[dduID] = (int)(dduHeader.lvl1num());
    int L1A_inc = L1ANumbers[dduID] - L1ANumber_previous_event;
    if (MEEMU("All_DDUs_L1A_Increment", me)) {
      if (L1A_inc > 100000) L1A_inc = 19;
      else if (L1A_inc > 30000) L1A_inc = 18;
      else if (L1A_inc > 10000) L1A_inc = 17;
      else if (L1A_inc > 3000)  L1A_inc = 16;
      else if (L1A_inc > 1000)  L1A_inc = 15;
      else if (L1A_inc > 300)   L1A_inc = 14;
      else if (L1A_inc > 100)   L1A_inc = 13;
      else if (L1A_inc > 30)    L1A_inc = 12;
      else if (L1A_inc > 10)    L1A_inc = 11;
      me->Fill(dduID, L1A_inc);
    }
  }

  int dmb_dav_header      = dduHeader.dmb_dav();
  int ddu_connected_inputs= dduHeader.live_cscs();
  int csc_error_state     = dduTrailer.dmb_full()&0x7FFF; // Only 15 inputs for DDU
  int csc_warning_state   = dduTrailer.dmb_warn()&0x7FFF; // Only 15 inputs for DDU
  int dmb_dav_header_cnt  = 0;
  int ddu_connected_inputs_cnt = 0;

  for (int i=0; i<15; ++i) {

    if ((dmb_dav_header>>i) & 0x1) {
      dmb_dav_header_cnt++;
      if (MEEMU("All_DDUs_Inputs_with_Data", me)) me->Fill(dduID, i);
    }

    if( (ddu_connected_inputs>>i) & 0x1 ){
      ddu_connected_inputs_cnt++;
      if (MEEMU("All_DDUs_Live_Inputs", me)) me->Fill(dduID, i);
    }

    if( (csc_error_state>>i) & 0x1 ){
      if (MEEMU("All_DDUs_Inputs_Errors", me)) me->Fill(dduID, i+2);
    }

    if( (csc_warning_state>>i) & 0x1 ){
      if (MEEMU("All_DDUs_Inputs_Warnings", me)) me->Fill(dduID, i+2);
    }

  }

  if (MEEMU("All_DDUs_Average_Live_Inputs", me)) me->Fill(dduID, ddu_connected_inputs_cnt);

  if (MEEMU("All_DDUs_Average_Inputs_with_Data", me)) me->Fill(dduID, dmb_dav_header_cnt);

  if (MEEMU("All_DDUs_Inputs_Errors", me)) {
    if (csc_error_state > 0) me->Fill(dduID, 1);   // Any Input
    else me->Fill(dduID, 0);                       // No errors
  }

  if (MEEMU("All_DDUs_Inputs_Warnings", me)) {
    if (csc_warning_state > 0) me->Fill(dduID, 1); // Any Input
    else me->Fill(dduID, 0);                       // No warnings
  }

  uint32_t trl_errorstat = dduTrailer.errorstat();
  if (dmb_dav_header_cnt==0) trl_errorstat &= ~0x20000000; // Ignore No Good DMB CRC bit of no DMB is present
  if (MEEMU("All_DDUs_Trailer_Errors", me)) {
    if (trl_errorstat) {
      me->Fill(dduID, 1); // Any Error
      for (int i=0; i<32; i++) {
        if ((trl_errorstat>>i) & 0x1) {
          me->Fill(dduID, i+2);
        }
      }
    } else {
      me->Fill(dduID, 0); // No Errors
    }
  }

  std::vector<CSCEventData> chamberDatas = dduEvent.cscData();

  int nCSCs = chamberDatas.size();
  if (nCSCs != dduHeader.ncsc()) {
    // == Current trick to maximize number of unpacked CSCs.
    // == Unpacker gives up after screwed chamber.
    // == So we need to exclude it from the list by reducing chamberDatas vector size
    nCSCs-=1;
    return;
  }

  for(unsigned int i=0; i < chamberDatas.size(); i++) {
    monitorCSC(chamberDatas[i], dduID);
  }

}
