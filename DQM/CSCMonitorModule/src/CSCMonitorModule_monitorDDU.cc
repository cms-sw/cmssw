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

#include <math.h>
#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"
#include "csc_utilities.cc"

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

  if (MEDDU(dduID, "Buffer_Size", me)) me->Fill(dduEvent.size());

  int trl_word_count = dduTrailer.wordcount();
  if (MEDDU(dduID, "Word_Count", me)) me->Fill(trl_word_count);
  if (trl_word_count > 0) {
    if (MEEMU("All_DDUs_Event_Size", me)) me->Fill(dduID, log10((double)trl_word_count));
  }

  if (MEEMU("All_DDUs_Average_Event_Size", me)) me->Fill(dduID, trl_word_count);

  uint32_t BXN=dduHeader.bxnum();
  if (MEDDU(dduID, "BXN", me)) me->Fill((double) BXN);

  std::map<uint32_t,uint32_t>::iterator it = L1ANumbers.find(dduID);
  if(it == L1ANumbers.end()) {
    L1ANumbers[dduID] = (int)(dduHeader.lvl1num());
  } else {

    int L1ANumber_previous_event = L1ANumbers[dduID];
    L1ANumbers[dduID] = (int)(dduHeader.lvl1num());
    int L1A_inc = L1ANumbers[dduID] - L1ANumber_previous_event;

    if (MEDDU(dduID, "L1A_Increment", me)) me->Fill(L1A_inc);

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
  int dmb_active_header   = (int)(dduHeader.ncsc()&0xF);
  int dmb_dav_header_cnt  = 0;
  int ddu_connected_inputs_cnt = 0;

  double freq = 0;
  for (int i=0; i<15; ++i) {

    if ((dmb_dav_header>>i) & 0x1) {
      dmb_dav_header_cnt++;
      if (MEDDU(dduID, "DMB_DAV_Header_Occupancy_Rate", me)) {
	me->Fill(i + 1);
	freq = (100.0 * me->getBinContent(i + 1)) / nEvents;
        if (MEDDU(dduID, "DMB_DAV_Header_Occupancy", me)) me->setBinContent(i + 1, freq);
      }
      if (MEEMU("All_DDUs_Inputs_with_Data", me)) me->Fill(dduID, i);
    }

    if( (ddu_connected_inputs>>i) & 0x1 ){
      ddu_connected_inputs_cnt++;
      if (MEDDU(dduID, "DMB_Connected_Inputs_Rate", me)) {
	me->Fill(i + 1);
	freq = (100.0 * me->getBinContent(i + 1)) / nEvents;
	if (MEDDU(dduID, "DMB_Connected_Inputs", me)) me->setBinContent(i + 1, freq);
      }
      if (MEEMU("All_DDUs_Live_Inputs", me)) me->Fill(dduID, i);
    }

    if( (csc_error_state>>i) & 0x1 ){
      if (MEDDU(dduID, "CSC_Errors_Rate", me)) {
	me->Fill(i + 1);
	freq = (100.0 * me->getBinContent(i + 1)) / nEvents;
	if (MEDDU(dduID, "CSC_Errors", me)) me->setBinContent(i + 1, freq);
      }
      if (MEEMU("All_DDUs_Inputs_Errors", me)) me->Fill(dduID, i+2);
    }

    if( (csc_warning_state>>i) & 0x1 ){
      if (MEDDU(dduID, "CSC_Warnings_Rate", me)) {
	me->Fill(i + 1);
	freq = (100.0 * me->getBinContent(i + 1)) / nEvents;
	if (MEDDU(dduID, "CSC_Warnings", me)) me->setBinContent(i + 1, freq);
      }
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

  if (MEDDU(dduID,"DMB_DAV_Header_Occupancy",me)) me->setEntries(nEvents);

  if (MEDDU(dduID, "DMB_Connected_Inputs", me)) me->setEntries(nEvents);

  if (MEDDU(dduID, "CSC_Errors", me)) me->setEntries(nEvents);

  if (MEDDU(dduID, "CSC_Warnings", me)) me->setEntries(nEvents);

  if (MEDDU(dduID, "DMB_Active_Header_Count", me)) me->Fill(dmb_active_header);

  if (MEDDU(dduID, "DMB_DAV_Header_Count_vs_DMB_Active_Header_Count", me)) me->Fill(dmb_active_header, dmb_dav_header_cnt);

  uint32_t trl_errorstat = dduTrailer.errorstat();
  if (dmb_dav_header_cnt==0) trl_errorstat &= ~0x20000000; // Ignore No Good DMB CRC bit of no DMB is present
  for (int i=0; i<32; i++) {
    if ((trl_errorstat>>i) & 0x1) {
      if (MEDDU(dduID, "Trailer_ErrorStat_Rate", me)) { 
	me->Fill(i);
	double freq = (100.0 * me->getBinContent(i + 1)) / nEvents;
	if (MEDDU(dduID, "Trailer_ErrorStat_Frequency", me)) me->setBinContent(i + 1, freq);
      }
      if (MEDDU(dduID, "Trailer_ErrorStat_Table", me)) me->Fill(0., i);
    }
  }
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

  if (MEDDU(dduID, "Trailer_ErrorStat_Table", me)) me->setEntries(nEvents);

  if (MEDDU(dduID, "Trailer_ErrorStat_Frequency", me)) me->setEntries(nEvents);

  std::vector<CSCEventData> chamberDatas = dduEvent.cscData();

  int nCSCs = chamberDatas.size();
  if (nCSCs != dduHeader.ncsc()) {
    // == Current trick to maximize number of unpacked CSCs.
    // == Unpacker gives up after screwed chamber.
    // == So we need to exclude it from the list by reducing chamberDatas vector size
    nCSCs-=1;
    return;
  }

  uint32_t unpackedDMBcount = 0;

  for(unsigned int i=0; i < chamberDatas.size(); i++) {
    unpackedDMBcount++;
    monitorCSC(chamberDatas[i], dduID);
  }

  if (MEDDU(dduID, "DMB_unpacked_vs_DAV", me)) me->Fill(dmb_active_header, unpackedDMBcount);

}
