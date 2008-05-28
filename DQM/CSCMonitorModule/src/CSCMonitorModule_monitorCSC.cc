/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule_monitorCSC.cc
 *
 *    Description:  Chamber monitor method implementation
 *
 *        Version:  1.0
 *        Created:  04/18/2008 03:39:49 PM
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
 * @brief  MonitorCSC function that grabs CSCEventData and processes it.
 * @param  cscData CSC data
 * @param  dduID DDU ID
 * @return
 */
void CSCMonitorModule::monitorCSC(const CSCEventData& cscEvent, const int32_t& dduID){

  MonitorElement* me;

  // Unpacking DMB header and trailer
  const CSCDMBHeader* dmbHeader = cscEvent.dmbHeader();
  const CSCDMBTrailer* dmbTrailer = cscEvent.dmbTrailer();
  if (!dmbHeader || !dmbTrailer) return;

  // Unpacking of Chamber Identification number
  int crateID	= dmbHeader->crateID();
  int dmbID	= dmbHeader->dmbID();
  // int ChamberID	= (((crateID) << 4) + dmbID) & 0xFFF;
  if (crateID == 0 || dmbID == 0) return;

  bool L1A_out_of_sync = false;

  // To be switched on on CSC enter :) 
  // int iendcap  = -1;
  // int istation = -1;
  // int icfeb    = 0;
  // int ilayer   = 0;
  // CSCDetId cid = pcrate->detId( crateID, dmbID, icfeb, ilayer);
  // int id = cscMapping.chamber(iendcap, istation, crateID, dmbID, -1);
  // if (id == 0) return;
  // CSCDetId cid(id);
  
  int CSCtype = 0;
  int CSCposition = 0;
  getCSCFromMap(crateID, dmbID, CSCtype, CSCposition );

  // Top level L1A synchronization
  int dmbHeaderL1A = dmbHeader->l1a()%64;
  if ((int)(dmbHeaderL1A - (int)(L1ANumber%64)) != 0) L1A_out_of_sync = true;

  if (CSCtype && CSCposition && MEEMU("CSC_Unpacked", me)) me->Fill(CSCposition, CSCtype);

  if (MEEMU("DMB_Unpacked", me)) me->Fill(crateID, dmbID);

  // 
  // Check if any of input FIFO's are full
  //

  bool anyInputFull = dmbTrailer->tmb_full || dmbTrailer->alct_full;
  for (int i=0; i<5; i++) {
    anyInputFull = anyInputFull || (int)((dmbTrailer->cfeb_full>>i)&0x1);
  }

  if(anyInputFull){
    if (CSCtype && CSCposition && MEEMU("CSC_DMB_input_fifo_full", me)) me->Fill(CSCposition, CSCtype);
    if (MEEMU("DMB_input_fifo_full", me)) me->Fill(crateID, dmbID);
  }

  //
  // Check if any input Timeouted
  //

  bool anyInputTO = dmbTrailer->tmb_timeout || dmbTrailer->alct_timeout || dmbTrailer->cfeb_starttimeout || dmbTrailer->cfeb_endtimeout;
  for (int i=0; i<5; i++) {
    anyInputTO = ((dmbTrailer->cfeb_starttimeout>>i) & 0x1) || ((dmbTrailer->cfeb_endtimeout>>i) & 0x1);
  }

  if(anyInputTO){
    if (CSCtype && CSCposition && MEEMU("CSC_DMB_input_timeout", me)) me->Fill(CSCposition, CSCtype);
    if (MEEMU("DMB_input_timeout", me)) me->Fill(crateID, dmbID);
  }

  //
  // Check ALCT
  //

  if (!cscEvent.nalct()) {
    if (CSCtype && CSCposition && MEEMU("CSC_wo_ALCT", me)) me->Fill(CSCposition, CSCtype);
    if (MEEMU("DMB_wo_ALCT", me)) me->Fill(crateID, dmbID);
  } else {
    if (cscEvent.alctHeader()) {
      const CSCALCTHeader* alctHeader = cscEvent.alctHeader();
      if ((int)(alctHeader->L1Acc()%64 - dmbHeaderL1A) != 0) L1A_out_of_sync = true;
    }
  }

  //
  // Check CLCT
  //

  if(!cscEvent.nclct()) {
    if (CSCtype && CSCposition && MEEMU("CSC_wo_CLCT", me)) me->Fill(CSCposition, CSCtype);
    if (MEEMU("DMB_wo_CLCT", me)) me->Fill(crateID, dmbID);
  }

  
  //
  // Checking CFEBs
  //
 
  int NumberOfUnpackedCFEBs = 0;
  const int N_CFEBs  = 5;
  const int N_Layers = 6;

  CSCCFEBData* cfebData[5];
  CSCCFEBTimeSlice* timeSlice[5][16];

  for(int nCFEB = 0; nCFEB < N_CFEBs; ++nCFEB) {

    cfebData[nCFEB] = cscEvent.cfebData(nCFEB);

    if (cfebData[nCFEB] !=0) {

      if (!cfebData[nCFEB]->check()) continue;

      NumberOfUnpackedCFEBs++;
      int NmbTimeSamples= (cfebData[nCFEB])->nTimeSamples();

      for(int nLayer = 1; nLayer <= N_Layers; ++nLayer) {
        for(int nSample = 0; nSample < NmbTimeSamples; ++nSample) {

          timeSlice[nCFEB][nSample] = (CSCCFEBTimeSlice * )((cfebData[nCFEB])->timeSlice(nSample));
          if ((int)((timeSlice[nCFEB][nSample]->get_L1A_number()) - dmbHeaderL1A) != 0) {
            L1A_out_of_sync = true;
          }

        }
      }

    }
  }

  if(NumberOfUnpackedCFEBs == 0) {
    if (CSCtype && CSCposition && MEEMU("CSC_wo_CFEB", me)) me->Fill(CSCposition, CSCtype);
    if (MEEMU("DMB_wo_CFEB", me)) me->Fill(crateID,dmbID);
  }

  // Checking L1A out of sync occurancies
  if (cscEvent.nclct() && cscEvent.nalct()) {
    if (cscEvent.alctHeader()) {
      if (cscEvent.tmbData()) {
        CSCTMBData* tmbData = cscEvent.tmbData();
        if (tmbData->tmbHeader()) {
          CSCTMBHeader* tmbHeader = tmbData->tmbHeader();
          if (((int)(tmbHeader->L1ANumber()%64) - dmbHeaderL1A) != 0) L1A_out_of_sync = true;
        }
      }
    }
  }

  if (CSCtype && CSCposition && L1A_out_of_sync && MEEMU("CSC_L1A_out_of_sync", me)) {
    me->Fill(CSCposition, CSCtype);
  }

}
