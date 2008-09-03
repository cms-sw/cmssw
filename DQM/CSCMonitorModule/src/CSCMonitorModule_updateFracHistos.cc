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
#include <time.h>

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
  
  int dbg = 0;
  long dbgt0 = clock();
  if (MEEMU("CSC_Reporting", me1)) {

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();
    // Getting reference and reporting histograms for CSC_Reporting
    TH2* ref = dynamic_cast<TH2*>(me1->getRefRootObject());
    TH2* rep = dynamic_cast<TH2*>(me1->getTH1());
    if (ref) {
      summary.ReadReportingChambersRef(rep, ref, 
        effParameters.getUntrackedParameter<double>("threshold_cold", 0.1), 
        effParameters.getUntrackedParameter<double>("sigfail_cold"  , 5.0), 
        effParameters.getUntrackedParameter<double>("threshold_hot" , 0.1), 
        effParameters.getUntrackedParameter<double>("sigfail_hot"   , 2.0));
    } else {
      summary.ReadReportingChambers(rep, 1.0);
    }


LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();
    double threshold = effParameters.getUntrackedParameter<double>("threshold_err", 0.1);
    double sigfail   = effParameters.getUntrackedParameter<double>("sigfail_err", 5.0);

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    if (MEEMU("CSC_Format_Errors", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, FORMAT_ERR, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    if (MEEMU("CSC_L1A_out_of_sync", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, L1SYNC_ERR, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    if (MEEMU("CSC_DMB_input_fifo_full", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, FIFOFULL_ERR, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    if (MEEMU("CSC_DMB_input_timeout", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, INPUTTO_ERR, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    threshold = effParameters.getUntrackedParameter<double>("threshold_nodata", 0.9);
    sigfail   = effParameters.getUntrackedParameter<double>("sigfail_nodata", 5.0);

    if (MEEMU("CSC_wo_ALCT", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, NODATA_ALCT, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    if (MEEMU("CSC_wo_CLCT", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, NODATA_CLCT, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    if (MEEMU("CSC_wo_CFEB", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, NODATA_CFEB, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

    if (MEEMU("CSC_Format_Warnings", me2)) {
      TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, CFEB_BWORDS, threshold, sigfail);
    }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  }

  //
  // Write Global DQM shifter chamber error maps 
  //
   
  if (MEEventInfo("reportSummaryMap", me1)) {
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x1, 3, true, false);
    summary.WriteChamberState(tmp, HWSTATUSERRORBITS, 2, false, true);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_occupancy", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x4, 4, true, false);
    summary.WriteChamberState(tmp, 0x8, 1, false, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_format_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x10, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_l1sync_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x20, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_fifofull_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x40, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_inputto_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x80, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_wo_alct", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x100, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_wo_clct", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x200, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_wo_cfeb", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x400, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("CSC_STATS_cfeb_bwords", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x800, 2, true, false);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  //
  // Write summary information
  //

  if (MEEMU("Physics_ME1", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 1);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0 << ", buvo > 100K";
dbgt0 = clock();

  if (MEEMU("Physics_ME2", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 2);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0 << ", buvo > 100K";
dbgt0 = clock();

  if (MEEMU("Physics_ME3", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 3);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0 << ", buvo > 100K";
dbgt0 = clock();

  if (MEEMU("Physics_ME4", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 4);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

  if (MEEMU("Physics_EMU", me1)) {
    TH2* tmp=dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteMap(tmp);
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0 << ", buvo > 2M";
dbgt0 = clock();

  // Looping via addresses (scope: side->station->ring) and
  // filling in HW efficiencies
  CSCAddress adr;
  adr.mask.side = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;

  if(MEEventInfo("reportSummary", me1))
    me1->Fill(summary.GetEfficiencyHW(adr));

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

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

      if (summary.Detector().NumberOfRings(adr.station) > 1) {

        adr.mask.ring = true;
        for (adr.ring = 1; adr.ring <= summary.Detector().NumberOfRings(adr.station); adr.ring++) {

          if(MEReportSummaryContents(summary.Detector().AddressName(adr), me1)) 
            me1->Fill(summary.GetEfficiencyHW(adr));

        }

      }
    }
  }

LOGINFO("debug") << "#" << dbg++ << ", elapsed = " << clock() - dbgt0;
dbgt0 = clock();

}

