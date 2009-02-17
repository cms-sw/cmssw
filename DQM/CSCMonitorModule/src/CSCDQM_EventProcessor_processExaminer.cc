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

#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"

namespace cscdqm {

  /**
   * @brief  Execute Examiner on Data buffer and collect output results.
   * @param  data Data buffer
   * @param  dataSize Data buffer size
   * @param  eventDenied Return flag if this buffer (event) was denied by Examiner or not
   * @return 
   */
  void EventProcessor::processExaminer(const uint16_t *data, const uint32_t dataSize, bool& eventDenied) {
    
    binChecker.setMask(config->getBINCHECK_MASK());
    
    if (binChecker.check(data, dataSize) < 0) {

      /** No ddu trailer found - force checker to summarize errors by adding artificial trailer */

      const uint16_t dduTrailer[4] = { 0x8000, 0x8000, 0xFFFF, 0x8000 };
      const uint16_t *tmp = dduTrailer;
      binChecker.check(tmp, uint32_t(4));
    }

    uint32_t binErrorStatus = binChecker.errors();
    uint32_t binWarningStatus = binChecker.warnings();

    MonitorObject* mo = 0;
    if (getEMUHisto(h::EMU_ALL_DDUS_FORMAT_ERRORS, mo)) {

      std::vector<int> DDUs = binChecker.listOfDDUs();
      for (std::vector<int>::iterator ddu_itr = DDUs.begin(); ddu_itr != DDUs.end(); ++ddu_itr) {
        if (*ddu_itr != 0xFFF) {
          long errs = binChecker.errorsForDDU(*ddu_itr);
          int dduID = (*ddu_itr)&0xFF;
          if (errs != 0) {
            for(int i = 0; i < binChecker.nERRORS; i++) { 
              if ((errs >> i) & 0x1 ) {
                mo->Fill(dduID, i + 1);
              }
            }
          }
        }
      }

      /** Temporary tweak for cases when there were no DDU errors  */

      if (binChecker.errors() == 0) {
        int dduID = binChecker.dduSourceID() & 0xFF;
        mo->Fill(dduID, 0);
      }

    }
  	
    if ((binErrorStatus & config->getDDU_BINCHECK_MASK()) > 0) {
      eventDenied = true;
    }

    if ( (binErrorStatus != 0) || (binWarningStatus != 0) ) {
      config->incNEventsBad();
    }

    std::map<int,long> payloads = binChecker.payloadDetailed();

    for(std::map<int,long>::const_iterator chamber=payloads.begin(); chamber!=payloads.end(); chamber++) {

      int chamberID = chamber->first;
      int crateID = (chamberID >> 4) & 0xFF;
      int dmbSlot = chamberID & 0xF;
      
      if (crateID == 255) { continue; }

      /** Update counters */
      config->incChamberCounter(DMB_EVENTS, crateID, dmbSlot);
      long DMBEvents = config->getChamberCounterValue(DMB_EVENTS, crateID, dmbSlot);
      config->copyChamberCounterValue(DMB_EVENTS, DMB_TRIGGERS, crateID, dmbSlot);

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
      long payload = chamber->second;
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

  /**  === Check and fill CSC Data Flow Problems */
   
  std::map<int,long> statuses = binChecker.statusDetailed();
  for(std::map<int,long>::const_iterator chamber = statuses.begin(); chamber != statuses.end(); chamber++) {

      int chamberID = chamber->first;

      unsigned int crateID = (chamberID >> 4) & 0xFF;
      unsigned int dmbSlot = chamberID & 0xF;

      if (crateID == 255) { continue; }

      unsigned int cscType   = 0;
      unsigned int cscPosition = 0;
      if (!getCSCFromMap(crateID, dmbSlot, cscType, cscPosition)) continue;

      if (getCSCHisto(h::CSC_BINCHECK_DATAFLOW_PROBLEMS_TABLE, crateID, dmbSlot, mo)) {
	for (int bit = 0; bit < binChecker.nSTATUSES; bit++) {
	  if (chamber->second & (1<<bit) ) {
	    mo->Fill(0., bit);
	  }
        }
	mo->SetEntries(config->getChamberCounterValue(DMB_EVENTS, crateID, dmbSlot));
      }

      
      int anyInputFull = chamber->second & 0x3F;
      if (anyInputFull) {
	if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_DMB_INPUT_FIFO_FULL, mo)) {
	  mo->Fill(cscPosition, cscType);
	}
	if (getEMUHisto(h::EMU_DMB_INPUT_FIFO_FULL, mo)) {
	  mo->Fill(crateID, dmbSlot);
	}
      }


      int anyInputTO = (chamber->second >> 7) & 0x3FFF;
      if (anyInputTO) {
	if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_DMB_INPUT_TIMEOUT, mo)) {
	  mo->Fill(cscPosition, cscType);
	}
	if (getEMUHisto(h::EMU_DMB_INPUT_TIMEOUT, mo)) {
	  mo->Fill(crateID, dmbSlot);
	}
      }
      
      if (chamber->second & (1 << 22)) {
	if (getEMUHisto(h::EMU_DMB_FORMAT_WARNINGS, mo)) {
	  mo->Fill(crateID, dmbSlot);
	}
  
	if (cscType && cscPosition && getEMUHisto(h::EMU_CSC_FORMAT_WARNINGS, mo)) {
	  mo->Fill(cscPosition, cscType);
	}
	
      }
    }

    /**  Check and fill CSC Format Errors  */
    std::map<int,long> checkerErrors = binChecker.errorsDetailed();
    for(std::map<int,long>::const_iterator chamber = checkerErrors.begin(); chamber != checkerErrors.end(); chamber++) {

      unsigned int chamberID = chamber->first;
      unsigned int crateID = (chamberID >> 4) & 0xFF;
      unsigned int dmbSlot = chamberID & 0xF;

      if ((crateID ==255) || 
	  (chamber->second & 0x80)) { continue; } // = Skip chamber detection if DMB header is missing (Error code 6)

      if (crateID > 60 || dmbSlot > 10) {
	continue;
      }
 
      if ((chamber->second & config->getBINCHECK_MASK()) != 0) {
        config->incChamberCounter(BAD_EVENTS, crateID , dmbSlot);
      }

      bool isCSCError = false;

      if (getCSCHisto(h::CSC_BINCHECK_ERRORSTAT_TABLE, crateID, dmbSlot, mo)) {
	for(int bit = 5; bit < 24; bit++) {
	  if( chamber->second & (1 << bit) ) {
	    isCSCError = true;
	    mo->Fill(0., bit - 5);
	  }
          mo->SetEntries(config->getChamberCounterValue(DMB_EVENTS, crateID , dmbSlot));
        }
      }

      if (isCSCError) {

	if (getEMUHisto(h::EMU_DMB_FORMAT_ERRORS, mo)) {
	  mo->Fill(crateID, dmbSlot);
	}

	if (!eventDenied  && getEMUHisto(h::EMU_DMB_UNPACKED_WITH_ERRORS, mo)) {
	  mo->Fill(crateID, dmbSlot);
	}

	unsigned int cscType   = 0;
	unsigned int cscPosition = 0;
        if (!getCSCFromMap(crateID, dmbSlot, cscType, cscPosition)) continue;

	if ( cscType && cscPosition && getEMUHisto(h::EMU_CSC_FORMAT_ERRORS, mo)) {
	  mo->Fill(cscPosition, cscType);
	}

	if (!eventDenied  && cscType && cscPosition && getEMUHisto(h::EMU_CSC_UNPACKED_WITH_ERRORS, mo)) {
	  mo->Fill(cscPosition, cscType);
	}
      }

    }

  }


}
