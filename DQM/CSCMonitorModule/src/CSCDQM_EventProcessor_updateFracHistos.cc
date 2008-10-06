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

#include "DQM/CSCMonitorModule/interface/CSCDQM_EventProcessor.h"

namespace cscdqm {

  void EventProcessor::updateFractionHistos() {

    calcEMUFractionHisto(EMU_DMB_FORMAT_ERRORS_FRACT, EMU_DMB_REPORTING, EMU_DMB_FORMAT_ERRORS);
    calcEMUFractionHisto(EMU_CSC_FORMAT_ERRORS_FRACT, EMU_CSC_REPORTING, EMU_CSC_FORMAT_ERRORS);
    calcEMUFractionHisto(EMU_DMB_FORMAT_WARNINGS_FRACT, EMU_DMB_REPORTING, EMU_DMB_FORMAT_WARNINGS);
    calcEMUFractionHisto(EMU_CSC_FORMAT_WARNINGS_FRACT, EMU_CSC_REPORTING, EMU_CSC_FORMAT_WARNINGS);
    calcEMUFractionHisto(EMU_DMB_UNPACKED_FRACT, EMU_DMB_REPORTING, EMU_DMB_UNPACKED);
    calcEMUFractionHisto(EMU_CSC_UNPACKED_FRACT, EMU_CSC_REPORTING, EMU_CSC_UNPACKED);
    calcEMUFractionHisto(EMU_DMB_WO_ALCT_FRACT, EMU_DMB_REPORTING, EMU_DMB_WO_ALCT);
    calcEMUFractionHisto(EMU_CSC_WO_ALCT_FRACT, EMU_CSC_REPORTING, EMU_CSC_WO_ALCT);
    calcEMUFractionHisto(EMU_DMB_WO_CLCT_FRACT, EMU_DMB_REPORTING, EMU_DMB_WO_CLCT);
    calcEMUFractionHisto(EMU_CSC_WO_CLCT_FRACT, EMU_CSC_REPORTING, EMU_CSC_WO_CLCT);
    calcEMUFractionHisto(EMU_DMB_WO_CFEB_FRACT, EMU_DMB_REPORTING, EMU_DMB_WO_CFEB);
    calcEMUFractionHisto(EMU_CSC_WO_CFEB_FRACT, EMU_CSC_REPORTING, EMU_CSC_WO_CFEB);
    calcEMUFractionHisto(EMU_CSC_DMB_INPUT_FIFO_FULL_FRACT, EMU_CSC_REPORTING, EMU_CSC_DMB_INPUT_FIFO_FULL);
    calcEMUFractionHisto(EMU_DMB_INPUT_FIFO_FULL_FRACT, EMU_DMB_REPORTING, EMU_DMB_INPUT_FIFO_FULL);
    calcEMUFractionHisto(EMU_CSC_DMB_INPUT_TIMEOUT_FRACT, EMU_CSC_REPORTING, EMU_CSC_DMB_INPUT_TIMEOUT);
    calcEMUFractionHisto(EMU_DMB_INPUT_TIMEOUT_FRACT, EMU_DMB_REPORTING, EMU_DMB_INPUT_TIMEOUT);
    calcEMUFractionHisto(EMU_CSC_L1A_OUT_OF_SYNC_FRACT, EMU_CSC_REPORTING, EMU_CSC_L1A_OUT_OF_SYNC);
    calcEMUFractionHisto(EMU_DMB_L1A_OUT_OF_SYNC_FRACT, EMU_DMB_REPORTING, EMU_DMB_L1A_OUT_OF_SYNC);

    unsigned int iter = 0, crateId, dmbId;
    METype *mo = 0, *mof = 0;
    while (histoProvider->nextCSC(iter, crateId, dmbId)) {

      std::string cscTag(Form("CSC_%03d_%02d", crateId, dmbId));

      if (getCSCHisto(crateId, dmbId, CSC_BINCHECK_DATAFLOW_PROBLEMS_TABLE, mo) && 
          getCSCHisto(crateId, dmbId, CSC_BINCHECK_DATAFLOW_PROBLEMS_FREQUENCY, mof)) {
        mof->getObject()->Reset();
        mof->getObject()->Add(mo->getObject());
        mof->getObject()->Scale(1. / (nDMBEvents[cscTag]));
        mof->getObject()->SetMaximum(1.);
        mof->SetEntries(nDMBEvents[cscTag]);
        mo->SetEntries(nDMBEvents[cscTag]);
      }

      if (getCSCHisto(crateId, dmbId, CSC_BINCHECK_ERRORSTAT_TABLE, mo) && 
          getCSCHisto(crateId, dmbId, CSC_BINCHECK_ERRORS_FREQUENCY, mof)) {
        mof->getObject()->Reset();
        mof->getObject()->Add(mo->getObject());
        mof->getObject()->Scale(1. / (nDMBEvents[cscTag]));
        mof->getObject()->SetMaximum(1.);
        mof->SetEntries(nDMBEvents[cscTag]);
        mo->SetEntries(nDMBEvents[cscTag]);
      }

    }

  }

/**
 * @brief Calculate fractional histogram 
 * @param result Histogram to write results to
 * @param set Histogram of the set
 * @param subset Histogram of the subset
 * @return 
 */

  void EventProcessor::calcEMUFractionHisto(const HistoType result, const HistoType set, const HistoType subset) {

    METype *mo = 0, *mo1 = 0, *mo2 = 0;

    if (getEMUHisto(result, mo) && getEMUHisto(set, mo2) && getEMUHisto(subset, mo1)) {
      mo->getObject()->Reset();
      mo->getObject()->Divide(mo1->getObject(), mo2->getObject());
      mo->getObject()->SetMaximum(1.);
    }

  }

}
