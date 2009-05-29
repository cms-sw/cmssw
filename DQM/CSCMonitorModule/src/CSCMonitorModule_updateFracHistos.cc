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

void CSCMonitorModule::updateFracHistos() {

  MonitorElement *me1 = NULL, *me2 = NULL, *me3 = NULL;

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

  if (MEEMU("CSC_L1A_out_of_sync_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_L1A_out_of_sync", me3)) 
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("DMB_Format_Warnings_Fract", me1) && MEEMU("DMB_Reporting", me2) && MEEMU("DMB_Format_Warnings", me3))
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  if (MEEMU("CSC_Format_Warnings_Fract", me1) && MEEMU("CSC_Reporting", me2) && MEEMU("CSC_Format_Warnings", me3))
    me1->getTH1()->Divide(me3->getTH1(), me2->getTH1());

  //
  // Set detector information
  //
  
  if (MEEMU("CSC_Reporting", me1)) {

    // Getting reference and reporting histograms for CSC_Reporting
    const TH2* ref = dynamic_cast<const TH2*>(me1->getRefRootObject());
    const TH2* rep = dynamic_cast<const TH2*>(me1->getTH1());
    if (ref) {
      summary.ReadReportingChambersRef(rep, ref, 
        effParameters.getUntrackedParameter<double>("threshold_cold", 0.1), 
        effParameters.getUntrackedParameter<double>("sigfail_cold"  , 5.0), 
        effParameters.getUntrackedParameter<double>("threshold_hot" , 0.1), 
        effParameters.getUntrackedParameter<double>("sigfail_hot"   , 2.0));
    } else {
      summary.ReadReportingChambers(rep, 1.0);
    }

    double threshold = effParameters.getUntrackedParameter<double>("threshold_err", 0.1);
    double sigfail   = effParameters.getUntrackedParameter<double>("sigfail_err", 5.0);

    if (MEEMU("CSC_Format_Errors", me2)) {
      const TH2* err = dynamic_cast<TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::FORMAT_ERR, threshold, sigfail);
    }

    if (MEEMU("CSC_L1A_out_of_sync", me2)) {
      const TH2* err = dynamic_cast<const TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::L1SYNC_ERR, threshold, sigfail);
    }

    if (MEEMU("CSC_DMB_input_fifo_full", me2)) {
      const TH2* err = dynamic_cast<const TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::FIFOFULL_ERR, threshold, sigfail);
    }

    if (MEEMU("CSC_DMB_input_timeout", me2)) {
      const TH2* err = dynamic_cast<const TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::INPUTTO_ERR, threshold, sigfail);
    }

    threshold = effParameters.getUntrackedParameter<double>("threshold_nodata", 1.0);
    sigfail   = effParameters.getUntrackedParameter<double>("sigfail_nodata", 5.0);

    if (MEEMU("CSC_wo_ALCT", me2)) {
      const TH2* err = dynamic_cast<const TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::NODATA_ALCT, threshold, sigfail);
    }

    if (MEEMU("CSC_wo_CLCT", me2)) {
      const TH2* err = dynamic_cast<const TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::NODATA_CLCT, threshold, sigfail);
    }

    if (MEEMU("CSC_wo_CFEB", me2)) {
      const TH2* err = dynamic_cast<const TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::NODATA_CFEB, threshold, sigfail);
    }

    if (MEEMU("CSC_Format_Warnings", me2)) {
      const TH2* err = dynamic_cast<const TH2*>(me2->getTH1());
      summary.ReadErrorChambers(rep, err, cscdqm::CFEB_BWORDS, threshold, sigfail);
    }
  }

  //
  // Write Global DQM shifter chamber error maps 
  //
   
  if (MEEventInfo("reportSummaryMap", me1)) {
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x1, 3, true, false);
    summary.WriteChamberState(tmp, HWSTATUSERRORBITS, 2, false, true);
  }

  if (MEEMU("CSC_STATS_occupancy", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x4, 2, true, false);
    summary.WriteChamberState(tmp, 0x8, 4, false, false);
  }

  if (MEEMU("CSC_STATS_format_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x10, 2, true, false);
  }

  if (MEEMU("CSC_STATS_l1sync_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x20, 2, true, false);
  }

  if (MEEMU("CSC_STATS_fifofull_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x40, 2, true, false);
  }

  if (MEEMU("CSC_STATS_inputto_err", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x80, 2, true, false);
  }

  if (MEEMU("CSC_STATS_wo_alct", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x100, 2, true, false);
  }

  if (MEEMU("CSC_STATS_wo_clct", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x200, 2, true, false);
  }

  if (MEEMU("CSC_STATS_wo_cfeb", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x400, 2, true, false);
  }

  if (MEEMU("CSC_STATS_cfeb_bwords", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteChamberState(tmp, 0x800, 2, true, false);
  }

  //
  // Write summary information
  //

  if (MEEMU("Physics_ME1", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 1);
  }

  if (MEEMU("Physics_ME2", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 2);
  }

  if (MEEMU("Physics_ME3", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 3);
  }

  if (MEEMU("Physics_ME4", me1)){
    TH2* tmp = dynamic_cast<TH2*>(me1->getTH1());
    summary.Write(tmp, 4);
  }

  if (MEEMU("Physics_EMU", me1)) {
    TH2* tmp=dynamic_cast<TH2*>(me1->getTH1());
    summary.WriteMap(tmp);
  }

  // Looping via addresses (scope: side->station->ring) and
  // filling in HW efficiencies
  cscdqm::Address adr;
  adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  adr.mask.side = true;

  double e_detector = 0, e_side = 0, e_station = 0, e_ring = 0;

  for (adr.side = 1; adr.side <= N_SIDES; adr.side++) {
    
    e_side = 0;
    adr.mask.station = true;

    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {

      e_station = 0;
      adr.mask.ring = true;

      for (adr.ring = 1; adr.ring <= summary.getDetector().NumberOfRings(adr.station); adr.ring++) {

        e_ring = summary.GetEfficiencyHW(adr);
        e_station += e_ring;

        if (summary.getDetector().NumberOfRings(adr.station) > 1) {

          if (MEReportSummaryContents(summary.getDetector().AddressName(adr), me1)) {

            me1->Fill(e_ring);

          }

        }

      }

      adr.mask.ring = false;
      e_station = e_station / summary.getDetector().NumberOfRings(adr.station);
      if (MEReportSummaryContents(summary.getDetector().AddressName(adr), me1)) me1->Fill(e_station);
      e_side += e_station;

    }

    adr.mask.station = false;
    e_side = e_side / N_STATIONS;
    if (MEReportSummaryContents(summary.getDetector().AddressName(adr), me1)) me1->Fill(e_side);
    e_detector += e_side; 

  }

  e_detector = e_detector / N_SIDES;
  if (MEEventInfo("reportSummary", me1)) me1->Fill(e_detector);

}

