/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModule_updateFracHistos.cc
 *
 *    Description:  Method updateFracHistos of CSCMonitorModule implementation.
 *    This method should be called after run or on demand.  
 *
 *        Version:  1.0
 *        Created:  04/23/2008 01:46:05 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"
#include "CSCUtilities.cc"

void CSCMonitorModule::updateFracHistos() {

  MonitorElement *me1 = NULL, *me2 = NULL, *me3 = NULL;

  //
  // Calculate Aggregate Histograms
  //

#ifdef CMSSW20

  if (MEEMU("DMB_Reporting", me1) && MEEMU("DMB_Format_Errors", me2) && MEEMU("DMB_Unpacked", me3)) {
    me1->getTH1()->Add(me2->getTH1(), me3->getTH1());
    if (MEEMU("DMB_Unpacked_with_errors", me2)) {
      me1->getTH1()->Add(me2->getTH1(), -1);
    }
  }

  if (MEEMU("CSC_Reporting", me1) && MEEMU("CSC_Format_Errors", me2) && MEEMU("CSC_Unpacked", me3)) {
    me1->getTH1()->Add(me2->getTH1(), me3->getTH1());
    if (MEEMU("CSC_Unpacked_with_errors", me2)) {
      me1->getTH1()->Add(me2->getTH1(), -1);
    }
  }
 
#endif  

  //
  // Calculate Fractional Histograms
  //

  if (MEEMU("DMB_Format_Errors_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_Format_Errors", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_Format_Errors_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_Format_Errors", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("DMB_Unpacked_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_Unpacked", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_Unpacked_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_Unpacked", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("DMB_wo_ALCT_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_wo_ALCT", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_wo_ALCT_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_wo_ALCT", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("DMB_wo_CLCT_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_wo_CLCT", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_wo_CLCT_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_wo_CLCT", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("DMB_wo_CFEB_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_wo_CFEB", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_wo_CFEB_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_wo_CFEB", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_DMB_input_fifo_full_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_DMB_input_fifo_full", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("DMB_input_fifo_full_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_input_fifo_full", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_DMB_input_timeout_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_DMB_input_timeout", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("DMB_input_timeout_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_input_timeout", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

#ifdef CMSSW21

  if (MEEMU("DMB_Format_Warnings_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_Format_Warnings", me3))
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_Format_Warnings_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_Format_Warnings", me3))
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

#endif

#ifdef CMSSW20

  if (MEEMU("DMB_Format_Warnings_Fract", me1) && MEEMU("DMB_Format_Warnings", me2) && MEEMU("DMB_Unpacked", me3)) {
    TH1* tmp=dynamic_cast<TH1*>(me3->getTH1()->Clone());
    tmp->Add(me2->getTH1());
    if (MEEMU("DMB_Unpacked_with_warnings", me3)) tmp->Add(me3->getTH1(), -1);
    me1->getTH1()->Divide(me2->getTH1(), tmp);
    delete tmp;
  }

  if (MEEMU("CSC_Format_Warnings_Fract", me1) && MEEMU("CSC_Format_Warnings", me2) && MEEMU("CSC_Unpacked", me3)) {
    TH1* tmp=dynamic_cast<TH1*>(me3->getTH1()->Clone());
    tmp->Add(me2->getTH1());
    if (MEEMU("CSC_Unpacked_with_warnings", me3)) tmp->Add(me3->getTH1(), -1);
    me1->getTH1()->Divide(me2->getTH1(), tmp);
    delete tmp;
  }

#endif

  //
  // Set detector information
  //

  if (MEEMU("CSC_Reporting", me1)) {

    // Getting reference and reporting histograms for CSC_Reporting
    TH2* ref = dynamic_cast<TH2*>(me1->getRefRootObject());
    TH2* rep = dynamic_cast<TH2*>(me1->getTH1());
    if (ref) {
      summary.ReadReportingChambersRef(rep, ref, 0.1, 5.0);
    } else {
      summary.ReadReportingChambers(rep, 1.0);
    }

    if (MEEMU("CSC_Format_Errors", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, 0.1, 5.0);
    }

    if (MEEMU("CSC_L1A_out_of_sync", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, 0.1, 5.0);
    }

    if (MEEMU("CSC_DMB_input_fifo_full", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, 0.1, 5.0);
    }

    if (MEEMU("CSC_DMB_input_timeout", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, 0.1, 5.0);
    }
  }

  //
  // Write summary information
  //

  if (MEEMU("Summary_ME1", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 1);
  }

  if (MEEMU("Summary_ME2", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 2);
  }

  if (MEEMU("Summary_ME3", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 3);
  }

  if (MEEMU("Summary_ME4", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 4);
  }

  if (MEEventInfo("reportSummaryMap", me1)){

    TH2* tmp=dynamic_cast<TH2*>(me1->getTH1());
    float rs = summary.WriteMap(tmp);
    TString title = Form("EMU Status: Physics Efficiency %.2f", rs);
    tmp->SetTitle(title);

    // Filling in the main summary number
    // Note: this uses a different approach then summary contents numbers
    // This one uses Physics efficinency
    if(MEEventInfo("reportSummary", me1))  me1->Fill(rs);

    // Looping via addresses (scope: side->station->ring) and
    // filling in HW efficiencies
    CSCAddress adr;
    adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
    adr.mask.side = true;
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) {
      adr.mask.station = adr.mask.ring = false;

      if(MEReportSummaryContents(summary.Detector().AddressName(adr), me1)) 
        me1->Fill(summary.GetEfficiencyHW(adr));

      adr.mask.station = true; 
      for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {
        adr.mask.ring = false;

        if(MEReportSummaryContents(summary.Detector().AddressName(adr), me1)) 
          me1->Fill(summary.GetEfficiencyHW(adr));

        adr.mask.ring = true;
        for (adr.ring = 1; adr.ring <= summary.Detector().NumberOfRings(adr.station); adr.ring++) {

          if(MEReportSummaryContents(summary.Detector().AddressName(adr), me1)) 
            me1->Fill(summary.GetEfficiencyHW(adr));

        }
      }
    }

  }

}

