/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor.cc
 *
 *    Description:  Process Examiner output
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

#include "CSCDQM_EventProcessor.h"

namespace cscdqm {

  /**
   * @brief  Fill monitor elements with CSCDCCFormatStatusDigi information.
   * @return true if this buffer (event) was accepted by Examiner else otherwise
   */
  bool EventProcessor::processExaminer(const CSCDCCExaminer& binChecker, const CSCDCCFormatStatusDigi& digi) {

    bool eventAccepted = true;
    MonitorObject* mo = 0;

    uint32_t binErrorStatus = digi.getDDUSummaryErrors();

    if (getEMUHisto(h::EMU_ALL_DDUS_FORMAT_ERRORS, mo)) {

      const std::set<DDUIdType> DDUs = digi.getListOfDDUs();
      for (std::set<DDUIdType>::const_iterator ddu_itr = DDUs.begin(); ddu_itr != DDUs.end(); ++ddu_itr) {
        ExaminerStatusType errs = digi.getDDUErrors(*ddu_itr);
        int dduID = (*ddu_itr)&0xFF;
        if (errs != 0) {
          for(int i = 0; i < 29; i++) { 
            if ((errs >> i) & 0x1 ) {
              mo->Fill(dduID, i + 1);
            }
          }
        } else {
          mo->Fill(dduID, 0);
        }
      }

    }
  
    // =VB= We want to use DCC level check mask as in CSCDCCUnpacker and not DDU mask
    // 	    Otherwise whole DCC event could be skipped because of a single chamber error
    unsigned long dccBinCheckMask = 0x06080016;
//    if ((binErrorStatus & config->getDDU_BINCHECK_MASK()) > 0) {
    if ((binErrorStatus & dccBinCheckMask) > 0) {
      eventAccepted = false;
    }

    if (binErrorStatus != 0) {
      config->incNEventsBad();
    }

    /** Check and fill CSC Payload information */

    {
      uint32_t i = 0;
      CSCIdType chamberID = 0;
      while (digi.nextCSCWithPayload(i, chamberID)) {

        int crateID = (chamberID >> 4) & 0xFF;
        int dmbSlot = chamberID & 0xF;
      
        if (crateID == 255) { continue; }

        // Check if in standby!
        { 
          CSCDetId cid;
          if (!config->fnGetCSCDetId(crateID, dmbSlot, cid)) {
            continue;
          } 
        }

        /** Update counters */
        config->incChamberCounter(DMB_EVENTS, crateID, dmbSlot);
        long DMBEvents = config->getChamberCounterValue(DMB_EVENTS, crateID, dmbSlot);
        config->copyChamberCounterValue(DMB_EVENTS, DMB_TRIGGERS, crateID, dmbSlot);
	cntDMBs++;

        if (getEMUHisto(h::EMU_DMB_REPORTING, mo)) {
          mo->Fill(crateID, dmbSlot);
        }

        unsigned int cscType   = 0;
        unsigned int cscPosition = 0;
        if (!getCSCFromMap(crateID, dmbSlot, cscType, cscPosition)) continue;

        if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_REPORTING, mo)) {
          mo->Fill(cscPosition, cscType);
        }

        /** Get FEBs Data Available Info */
        long payload = digi.getCSCPayload(chamberID);
        int cfeb_dav = (payload >> 7) & 0x1F;
        int cfeb_active = payload & 0x1F;
        int alct_dav = (payload >> 5) & 0x1;
        int tmb_dav = (payload >> 6) & 0x1; 
        int cfeb_dav_num = 0;
      
        if (alct_dav == 0) {
          if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_WO_ALCT, mo)) {
            mo->Fill(cscPosition, cscType);
          }
          if (getEMUHisto(h::EMU_DMB_WO_ALCT, mo)) {
            mo->Fill(crateID, dmbSlot);
          }
        }
     
        if (tmb_dav == 0) {
          if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_WO_CLCT, mo)) {
            mo->Fill(cscPosition, cscType);
          }
          if (getEMUHisto(h::EMU_DMB_WO_CLCT, mo)) {
            mo->Fill(crateID, dmbSlot);
          }
        }

        if (cfeb_dav == 0) {
          if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_WO_CFEB, mo)) {
            mo->Fill(cscPosition, cscType);
          }
          if (getEMUHisto(h::EMU_DMB_WO_CFEB, mo)) {
            mo->Fill(crateID, dmbSlot);
          }
        }

	/** Increment total number of CFEBs, ALCTs, TMBs **/
        for (int i=0; i<5;i++) {
            if ((cfeb_dav>>i) & 0x1) cntCFEBs++;
        }
	
        if (alct_dav > 0) {
            cntALCTs++;
        }

        if (tmb_dav > 0) {
            cntTMBs++;
        }

 
        MonitorObject *mof = 0, *mo1 = 0, *mo2 = 0;
        if (getCSCHisto(h::CSC_ACTUAL_DMB_CFEB_DAV_RATE, crateID, dmbSlot, mo)
          && getCSCHisto(h::CSC_ACTUAL_DMB_CFEB_DAV_FREQUENCY, crateID, dmbSlot, mof)) {
          if (getCSCHisto(h::CSC_DMB_CFEB_DAV_UNPACKING_INEFFICIENCY, crateID, dmbSlot, mo1)
            && getCSCHisto(h::CSC_DMB_CFEB_DAV, crateID, dmbSlot, mo2)) {
            for (int i=1; i<=5; i++) {
              double actual_dav_num = mo->GetBinContent(i);
              double unpacked_dav_num = mo2->GetBinContent(i);
              if (actual_dav_num){
                mo1->SetBinContent(i,1, 100.*(1-unpacked_dav_num/actual_dav_num));
              }				   
              mo1->SetEntries((int)DMBEvents);
            }
          }	
          for (int i=0; i<5;i++) {
            int cfeb_present = (cfeb_dav>>i) & 0x1;
            cfeb_dav_num += cfeb_present;
            if (cfeb_present) {
              mo->Fill(i);
            }
            float cfeb_entries = mo->GetBinContent(i+1);
            mof->SetBinContent(i+1, ((float)cfeb_entries/(float)(DMBEvents)*100.0));
          }
          mof->SetEntries((int)DMBEvents);
        }
      
        if (getCSCHisto(h::CSC_ACTUAL_DMB_CFEB_DAV_MULTIPLICITY_RATE, crateID, dmbSlot, mo)
	  && getCSCHisto(h::CSC_ACTUAL_DMB_CFEB_DAV_MULTIPLICITY_FREQUENCY, crateID, dmbSlot, mof)) {
          for (unsigned short i = 1; i < 7; i++) {
            float cfeb_entries =  mo->GetBinContent(i);
            mof->SetBinContent(i, ((float)cfeb_entries / (float)(DMBEvents) * 100.0));
          }
          mof->SetEntries((int)DMBEvents);

          if (getCSCHisto(h::CSC_DMB_CFEB_DAV_MULTIPLICITY_UNPACKING_INEFFICIENCY, crateID, dmbSlot, mo1)
	    && getCSCHisto(h::CSC_DMB_CFEB_DAV_MULTIPLICITY, crateID, dmbSlot, mo2)) {	   
            for (unsigned short i = 1; i < 7; i++) {
              float actual_dav_num = mo->GetBinContent(i);
              float unpacked_dav_num = mo2->GetBinContent(i);
              if (actual_dav_num){
                mo1->SetBinContent(i, 1, 100. * (1-unpacked_dav_num/actual_dav_num));
              }				   
              mo1->SetEntries((int)DMBEvents);
            }
          }	
          mo->Fill(cfeb_dav_num);
        }

        if (getCSCHisto(h::CSC_DMB_CFEB_ACTIVE_VS_DAV, crateID, dmbSlot, mo)) mo->Fill(cfeb_dav, cfeb_active);

        /** Fill Histogram for FEB DAV Efficiency */
        if (getCSCHisto(h::CSC_ACTUAL_DMB_FEB_DAV_RATE, crateID, dmbSlot, mo)) {
          if (getCSCHisto(h::CSC_ACTUAL_DMB_FEB_DAV_FREQUENCY, crateID, dmbSlot, mo1)) {
            for (int i = 1; i < 4; i++) {
              float dav_num = mo->GetBinContent(i);
              mo1->SetBinContent(i, ((float)dav_num / (float)(DMBEvents) * 100.0));
            }
            mo1->SetEntries((int)DMBEvents);

            if (getCSCHisto(h::CSC_DMB_FEB_DAV_UNPACKING_INEFFICIENCY, crateID, dmbSlot, mof)
              && getCSCHisto(h::CSC_DMB_FEB_DAV_RATE, crateID, dmbSlot, mo2)) {	   
              for (int i = 1; i < 4; i++) {
                float actual_dav_num = mo->GetBinContent(i);
                float unpacked_dav_num = mo2->GetBinContent(i);
                if (actual_dav_num){
                  mof->SetBinContent(i,1, 100. * (1 - unpacked_dav_num / actual_dav_num));
                }				   
                mof->SetEntries((int)DMBEvents);
                mof->SetMaximum(100.0);
              }
            }	  
          }

          if (alct_dav > 0) {
            mo->Fill(0.0);
          }
          if (tmb_dav > 0) {
            mo->Fill(1.0);
          }
          if (cfeb_dav > 0) {
            mo->Fill(2.0);
          }
        }
      

        float feb_combination_dav = -1.0;
        /** Fill Histogram for Different Combinations of FEB DAV Efficiency */
        if (getCSCHisto(h::CSC_ACTUAL_DMB_FEB_COMBINATIONS_DAV_RATE, crateID, dmbSlot, mo)) {
          if(alct_dav == 0 && tmb_dav == 0 && cfeb_dav == 0) feb_combination_dav = 0.0; // Nothing
          if(alct_dav >  0 && tmb_dav == 0 && cfeb_dav == 0) feb_combination_dav = 1.0; // ALCT Only
          if(alct_dav == 0 && tmb_dav >  0 && cfeb_dav == 0) feb_combination_dav = 2.0; // TMB Only
          if(alct_dav == 0 && tmb_dav == 0 && cfeb_dav >  0) feb_combination_dav = 3.0; // CFEB Only
          if(alct_dav == 0 && tmb_dav >  0 && cfeb_dav >  0) feb_combination_dav = 4.0; // TMB+CFEB
          if(alct_dav >  0 && tmb_dav >  0 && cfeb_dav == 0) feb_combination_dav = 5.0; // ALCT+TMB
          if(alct_dav >  0 && tmb_dav == 0 && cfeb_dav >  0) feb_combination_dav = 6.0; // ALCT+CFEB
          if(alct_dav >  0 && tmb_dav >  0 && cfeb_dav >  0) feb_combination_dav = 7.0; // ALCT+TMB+CFEB
          // mo->Fill(feb_combination_dav);

          if (getCSCHisto(h::CSC_ACTUAL_DMB_FEB_COMBINATIONS_DAV_FREQUENCY, crateID, dmbSlot, mo1)) {
            for (int i = 1; i < 9; i++) {
              float feb_combination_dav_number = mo->GetBinContent(i);
              mo1->SetBinContent(i, ((float)feb_combination_dav_number / (float)(DMBEvents) * 100.0));
            }
            mo1->SetEntries(DMBEvents);
	  
            if (getCSCHisto(h::CSC_DMB_FEB_COMBINATIONS_DAV_UNPACKING_INEFFICIENCY, crateID, dmbSlot, mof)
     	      && getCSCHisto(h::CSC_DMB_FEB_COMBINATIONS_DAV_RATE, crateID, dmbSlot, mo2)) {	   
              for (int i = 1; i < 9; i++) {
                float actual_dav_num = mo->GetBinContent(i);
                float unpacked_dav_num = mo2->GetBinContent(i);
                if (actual_dav_num){
                  mof->SetBinContent(i, 1, 100. * (1 - unpacked_dav_num / actual_dav_num));
                }				   
                mof->SetEntries((int)DMBEvents);
                mof->SetMaximum(100.0);
              }
            }
	  
          }
          mo->Fill(feb_combination_dav);
        }
      
      }
    }

    /** Check and fill CSC Data Flow Problems */
   
    {
      uint32_t i = 0;
      CSCIdType chamberID = 0;
      while (digi.nextCSCWithStatus(i, chamberID)) {

        unsigned int crateID = (chamberID >> 4) & 0xFF;
        unsigned int dmbSlot = chamberID & 0xF;
        ExaminerStatusType chStatus = digi.getCSCStatus(chamberID);

        if (crateID == 255) { continue; }
         
        // Check if in standby!
        { 
          CSCDetId cid;
          if (!config->fnGetCSCDetId(crateID, dmbSlot, cid)) {
            continue;
          } 
        }

        unsigned int cscType   = 0;
        unsigned int cscPosition = 0;
        if (!getCSCFromMap(crateID, dmbSlot, cscType, cscPosition)) continue;

        if (getCSCHisto(h::CSC_BINCHECK_DATAFLOW_PROBLEMS_TABLE, crateID, dmbSlot, mo)) {
          for (int bit = 0; bit < binChecker.nSTATUSES; bit++) {
            if (chStatus & (1<<bit) ) {
              mo->Fill(0., bit);
            }
          }
          mo->SetEntries(config->getChamberCounterValue(DMB_EVENTS, crateID, dmbSlot));
        }

      
        int anyInputFull = chStatus & 0x3F;
        if (anyInputFull) {
          if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_DMB_INPUT_FIFO_FULL, mo)) {
            mo->Fill(cscPosition, cscType);
          }
          if (getEMUHisto(h::EMU_DMB_INPUT_FIFO_FULL, mo)) {
            mo->Fill(crateID, dmbSlot);
          }
        }

        int anyInputTO = (chStatus >> 7) & 0x3FFF;
        if (anyInputTO) {
          if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_DMB_INPUT_TIMEOUT, mo)) {
            mo->Fill(cscPosition, cscType);
          }
          if (getEMUHisto(h::EMU_DMB_INPUT_TIMEOUT, mo)) {
            mo->Fill(crateID, dmbSlot);
          }
        }
      
        if (digi.getCSCStatus(chamberID) & (1 << 22)) {
          if (getEMUHisto(h::EMU_DMB_FORMAT_WARNINGS, mo)) {
            mo->Fill(crateID, dmbSlot);
          }
  
          if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_FORMAT_WARNINGS, mo)) {
            mo->Fill(cscPosition, cscType);
          }
          
        }
      }

    }

    /**  Check and fill CSC Format Errors  */

    {
      uint32_t i = 0;
      CSCIdType chamberID = 0;
      while (digi.nextCSCWithError(i, chamberID)) {

        const unsigned int crateID = (chamberID >> 4) & 0xFF;
        const unsigned int dmbSlot = chamberID & 0xF;
        const ExaminerStatusType chErr = digi.getCSCErrors(chamberID);

        if ((crateID ==255) || (chErr & 0x80)) { continue; } // = Skip chamber detection if DMB header is missing (Error code 6)

        if (crateID > 60 || dmbSlot > 10) { continue; }
 
        // Check if in standby!
        { 
          CSCDetId cid;
          if (!config->fnGetCSCDetId(crateID, dmbSlot, cid)) {
            continue;
          } 
        }

        if ((chErr & config->getBINCHECK_MASK()) != 0) {
          config->incChamberCounter(BAD_EVENTS, crateID , dmbSlot);
        }

        bool isCSCError = false;
        bool fillBC = getCSCHisto(h::CSC_BINCHECK_ERRORSTAT_TABLE, crateID, dmbSlot, mo);

        for (int bit = 5; bit < 24; bit++) {

          if (chErr & (1 << bit) ) {
            isCSCError = true;
            if (fillBC) {
              mo->Fill(0., bit - 5);
            } else {
              break;
            }
          }

          if (fillBC) {
            mo->SetEntries(config->getChamberCounterValue(DMB_EVENTS, crateID , dmbSlot));
          }

        }

        if (isCSCError) {

          if (getEMUHisto(h::EMU_DMB_FORMAT_ERRORS, mo)) {
            mo->Fill(crateID, dmbSlot);
          }

          if (eventAccepted && getEMUHisto(h::EMU_DMB_UNPACKED_WITH_ERRORS, mo)) {
            mo->Fill(crateID, dmbSlot);
          }

          unsigned int cscType   = 0;
          unsigned int cscPosition = 0;
          if (!getCSCFromMap(crateID, dmbSlot, cscType, cscPosition)) continue;

          if ( cscType && cscPosition && getEMUHisto(h::EMU_CSC_FORMAT_ERRORS, mo)) {
            mo->Fill(cscPosition, cscType);
          }

          if (eventAccepted  && cscType && cscPosition && getEMUHisto(h::EMU_CSC_UNPACKED_WITH_ERRORS, mo)) {
            mo->Fill(cscPosition, cscType);
          }
        }

      }
    }

    return eventAccepted;

  }

}
