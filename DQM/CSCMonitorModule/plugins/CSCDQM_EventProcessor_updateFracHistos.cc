/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor_updateFracHistos.cc
 *
 *    Description:  Update Fractional and efficiency histograms
 *
 *        Version:  1.0
 *        Created:  10/06/2008 09:54:55 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "CSCDQM_EventProcessor.h"

namespace cscdqm {

  /**
   * @brief  Update Fractional MOs
   */
  void EventProcessor::updateFractionHistos() {

    calcEMUFractionHisto(h::EMU_DMB_FORMAT_ERRORS_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_FORMAT_ERRORS);
    calcEMUFractionHisto(h::EMU_CSC_FORMAT_ERRORS_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_FORMAT_ERRORS);
    calcEMUFractionHisto(h::EMU_DMB_FORMAT_WARNINGS_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_FORMAT_WARNINGS);
    calcEMUFractionHisto(h::EMU_CSC_FORMAT_WARNINGS_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_FORMAT_WARNINGS);
    calcEMUFractionHisto(h::EMU_DMB_UNPACKED_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_UNPACKED);
    calcEMUFractionHisto(h::EMU_CSC_UNPACKED_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_UNPACKED);
    calcEMUFractionHisto(h::EMU_DMB_WO_ALCT_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_WO_ALCT);
    calcEMUFractionHisto(h::EMU_CSC_WO_ALCT_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_WO_ALCT);
    calcEMUFractionHisto(h::EMU_DMB_WO_CLCT_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_WO_CLCT);
    calcEMUFractionHisto(h::EMU_CSC_WO_CLCT_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_WO_CLCT);
    calcEMUFractionHisto(h::EMU_DMB_WO_CFEB_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_WO_CFEB);
    calcEMUFractionHisto(h::EMU_CSC_WO_CFEB_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_WO_CFEB);
    calcEMUFractionHisto(h::EMU_CSC_DMB_INPUT_FIFO_FULL_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_DMB_INPUT_FIFO_FULL);
    calcEMUFractionHisto(h::EMU_DMB_INPUT_FIFO_FULL_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_INPUT_FIFO_FULL);
    calcEMUFractionHisto(h::EMU_CSC_DMB_INPUT_TIMEOUT_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_DMB_INPUT_TIMEOUT);
    calcEMUFractionHisto(h::EMU_DMB_INPUT_TIMEOUT_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_INPUT_TIMEOUT);
    calcEMUFractionHisto(h::EMU_CSC_L1A_OUT_OF_SYNC_FRACT, h::EMU_CSC_REPORTING, h::EMU_CSC_L1A_OUT_OF_SYNC);
    calcEMUFractionHisto(h::EMU_DMB_L1A_OUT_OF_SYNC_FRACT, h::EMU_DMB_REPORTING, h::EMU_DMB_L1A_OUT_OF_SYNC);
    calcEMUFractionHisto(h::EMU_FED_DDU_L1A_MISMATCH_FRACT, h::EMU_FED_ENTRIES, h::EMU_FED_DDU_L1A_MISMATCH);
    calcEMUFractionHisto(h::EMU_FED_DDU_L1A_MISMATCH_WITH_CSC_DATA_FRACT, h::EMU_FED_ENTRIES, h::EMU_FED_DDU_L1A_MISMATCH_WITH_CSC_DATA);

    unsigned int iter = 0, crateId = 0, dmbId = 0;
    MonitorObject *mo = 0, *mof = 0;
    while (config->fnNextBookedCSC(iter, crateId, dmbId)) {

      uint32_t dmbEvents = config->getChamberCounterValue(DMB_EVENTS, crateId, dmbId);

      if (getCSCHisto(h::CSC_BINCHECK_DATAFLOW_PROBLEMS_TABLE, crateId, dmbId, mo) && 
          getCSCHisto(h::CSC_BINCHECK_DATAFLOW_PROBLEMS_FREQUENCY, crateId, dmbId, mof)) {
        
        LockType lock(mof->mutex);
        TH1* th = mof->getTH1Lock();
        th->Reset();
        th->Add(mo->getTH1());
        th->Scale(1. / dmbEvents);
        mof->SetMaximum(1.);
        mof->SetEntries(dmbEvents);
        mo->SetEntries(dmbEvents);
      }

      if (getCSCHisto(h::CSC_BINCHECK_ERRORSTAT_TABLE, crateId, dmbId, mo) && 
          getCSCHisto(h::CSC_BINCHECK_ERRORS_FREQUENCY, crateId, dmbId, mof)) {
        LockType lock(mof->mutex);
        TH1* th = mof->getTH1Lock();
        th->Reset();
        th->Add(mo->getTH1());
        th->Scale(1. / dmbEvents);
        mof->SetMaximum(1.);
        mof->SetEntries(dmbEvents);
        mo->SetEntries(dmbEvents);
      }

    }

  }

  /**
   * @brief Calculate fractional histogram 
   * @param result Histogram to write results to
   * @param set Histogram of the set
   * @param subset Histogram of the subset
   */
  void EventProcessor::calcEMUFractionHisto(const HistoId& result, const HistoId& set, const HistoId& subset) {

    MonitorObject *mo = 0, *mo1 = 0, *mo2 = 0;

    if (getEMUHisto(result, mo) && getEMUHisto(set, mo2) && getEMUHisto(subset, mo1)) {
      LockType lock(mo->mutex);
      TH1* th = mo->getTH1Lock();
      th->Reset();
      th->Divide(mo1->getTH1(), mo2->getTH1());
      mo->SetMaximum(1.);
    }

  }

}
