/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor_updateEffHistos.cc
 *
 *    Description:  Update Efficiency histograms and parameters
 *
 *        Version:  1.0
 *        Created:  10/06/2008 11:44:34 AM
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
  
  void EventProcessor::updateEfficiencyHistos(EffParametersType& effParams) {

    METype *me = 0, *me1 = 0, *me2 = 0;

    if (getEMUHisto(EMU_CSC_REPORTING, me)) {

      TH2* rep = dynamic_cast<TH2*>(me->getTH1());

      // If reference for CSC_Reporting is defined - use it
      // Else - do it flat way
      if (getEMUHisto(EMU_CSC_REPORTING, me2, true)) {
        TH2* ref = dynamic_cast<TH2*>(me2->getTH1());
        summary.ReadReportingChambersRef(rep, ref, 
            effParams.cold_threshold, effParams.cold_sigfail, effParams.hot_threshold, effParams.hot_sigfail);
      } else {
        summary.ReadReportingChambers(rep, 1.0);
      }

      if (getEMUHisto(EMU_CSC_FORMAT_ERRORS, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, FORMAT_ERR, effParams.err_threshold, effParams.err_sigfail);
      }

      if (getEMUHisto(EMU_CSC_L1A_OUT_OF_SYNC, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, L1SYNC_ERR, effParams.err_threshold, effParams.err_sigfail);
      }

      if (getEMUHisto(EMU_CSC_DMB_INPUT_FIFO_FULL, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, FIFOFULL_ERR, effParams.err_threshold, effParams.err_sigfail);
      }

      if (getEMUHisto(EMU_CSC_DMB_INPUT_TIMEOUT, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, INPUTTO_ERR, effParams.err_threshold, effParams.err_sigfail);
      }

      if (getEMUHisto(EMU_CSC_WO_ALCT, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, NODATA_ALCT, effParams.nodata_threshold, effParams.nodata_sigfail);
      }

      if (getEMUHisto(EMU_CSC_WO_CLCT, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, NODATA_CLCT, effParams.nodata_threshold, effParams.nodata_sigfail);
      }

      if (getEMUHisto(EMU_CSC_WO_CFEB, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, NODATA_CFEB, effParams.nodata_threshold, effParams.nodata_sigfail);
      }

      if (getEMUHisto(EMU_CSC_FORMAT_WARNINGS, me1)) {
        TH2* err = dynamic_cast<TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, CFEB_BWORDS, effParams.nodata_threshold, effParams.nodata_sigfail);
      }

    }
    
    if (getEMUHisto(EMU_CSC_STATS_SUMMARY, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x1, 3, true, false);
      summary.WriteChamberState(tmp, HWSTATUSERRORBITS, 2, false, true);
    }

    if (getEMUHisto(EMU_CSC_STATS_OCCUPANCY, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x4, 2, true, false);
      summary.WriteChamberState(tmp, 0x8, 4, false, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_FORMAT_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x10, 2, true, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_L1SYNC_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x20, 2, true, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_FIFOFULL_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x40, 2, true, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_INPUTTO_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x80, 2, true, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_WO_ALCT, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x100, 2, true, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_WO_CLCT, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x200, 2, true, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_WO_CFEB, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x400, 2, true, false);
    }

    if (getEMUHisto(EMU_CSC_STATS_CFEB_BWORDS, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.WriteChamberState(tmp, 0x800, 2, true, false);
    }

  //
  // Write summary information
  //

    if (getEMUHisto(EMU_PHYSICS_ME1, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.Write(tmp, 1);
    }
  
    if (getEMUHisto(EMU_PHYSICS_ME2, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.Write(tmp, 2);
    }

    if (getEMUHisto(EMU_PHYSICS_ME3, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.Write(tmp, 3);
    }

    if (getEMUHisto(EMU_PHYSICS_ME4, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1());
      summary.Write(tmp, 4);
    }

    if (getEMUHisto(EMU_PHYSICS_EMU, me)) {
      TH2* tmp=dynamic_cast<TH2*>(me->getTH1());
      summary.WriteMap(tmp);
    }

    // Looping via addresses (scope: side->station->ring) and
    // filling in HW efficiencies
    Address adr;
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
            if (histoProvider->getEffParamHisto(summary.getDetector().AddressName(adr), me)) {
              me->Fill(e_ring);
            }
          }
        }
        adr.mask.ring = false;
        e_station = e_station / summary.getDetector().NumberOfRings(adr.station);
        if (histoProvider->getEffParamHisto(summary.getDetector().AddressName(adr), me)) me1->Fill(e_station);
        e_side += e_station;
      }
      adr.mask.station = false;
      e_side = e_side / N_STATIONS;
      if (histoProvider->getEffParamHisto(summary.getDetector().AddressName(adr), me)) me1->Fill(e_side);
      e_detector += e_side; 
    }
    e_detector = e_detector / N_SIDES;
    if (histoProvider->getEffParamHisto("reportSummary", me)) me1->Fill(e_detector);

  }

}

