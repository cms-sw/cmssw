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
  
  void EventProcessor::updateEfficiencyHistos() {

    MonitorObject *me = 0, *me1 = 0;

    if (getEMUHisto(h::EMU_CSC_REPORTING, me)) {

      const TH2* rep = dynamic_cast<const TH2*>(me->getTH1());

      // Get CSC Reporting reference histogram
      const TObject *tobj = me->getRefRootObject();
       
      // If reference for CSC_Reporting is defined - use it
      // Else - do it flat way
       
      if (tobj) {
        const TH2* ref = dynamic_cast<const TH2*>(tobj);
        summary.ReadReportingChambersRef(rep, ref, config->getEFF_COLD_THRESHOLD(), config->getEFF_COLD_SIGFAIL(), config->getEFF_HOT_THRESHOLD(), config->getEFF_HOT_SIGFAIL());
      } else {
        summary.ReadReportingChambers(rep, 1.0);
      }

      if (getEMUHisto(h::EMU_CSC_FORMAT_ERRORS, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, FORMAT_ERR, config->getEFF_ERR_THRESHOLD(), config->getEFF_ERR_SIGFAIL());
      }

      if (getEMUHisto(h::EMU_CSC_L1A_OUT_OF_SYNC, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, L1SYNC_ERR, config->getEFF_ERR_THRESHOLD(), config->getEFF_ERR_SIGFAIL());
      }

      if (getEMUHisto(h::EMU_CSC_DMB_INPUT_FIFO_FULL, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, FIFOFULL_ERR, config->getEFF_ERR_THRESHOLD(), config->getEFF_ERR_SIGFAIL());
      }

      if (getEMUHisto(h::EMU_CSC_DMB_INPUT_TIMEOUT, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, INPUTTO_ERR, config->getEFF_ERR_THRESHOLD(), config->getEFF_ERR_SIGFAIL());
      }

      if (getEMUHisto(h::EMU_CSC_WO_ALCT, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, NODATA_ALCT, config->getEFF_NODATA_THRESHOLD(), config->getEFF_NODATA_SIGFAIL());
      }

      if (getEMUHisto(h::EMU_CSC_WO_CLCT, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, NODATA_CLCT, config->getEFF_NODATA_THRESHOLD(), config->getEFF_NODATA_SIGFAIL());
      }

      if (getEMUHisto(h::EMU_CSC_WO_CFEB, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, NODATA_CFEB, config->getEFF_NODATA_THRESHOLD(), config->getEFF_NODATA_SIGFAIL());
      }

      if (getEMUHisto(h::EMU_CSC_FORMAT_WARNINGS, me1)) {
        const TH2* err = dynamic_cast<const TH2*>(me1->getTH1());
        summary.ReadErrorChambers(rep, err, CFEB_BWORDS, config->getEFF_NODATA_THRESHOLD(), config->getEFF_NODATA_SIGFAIL());
      }

    }

    if (getEMUHisto(h::EMU_CSC_STATS_SUMMARY, me)) {
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x1, 3, true, false);
      summary.WriteChamberState(tmp, HWSTATUSERRORBITS, 2, false, true);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_OCCUPANCY, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x4, 2, true, false);
      summary.WriteChamberState(tmp, 0x8, 4, false, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_FORMAT_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x10, 2, true, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_L1SYNC_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x20, 2, true, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_FIFOFULL_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x40, 2, true, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_INPUTTO_ERR, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x80, 2, true, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_WO_ALCT, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x100, 2, true, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_WO_CLCT, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x200, 2, true, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_WO_CFEB, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x400, 2, true, false);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_CSC_STATS_CFEB_BWORDS, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteChamberState(tmp, 0x800, 2, true, false);
      me->unlock();
    }
    
    //
    // Write summary information
    //

    if (getEMUHisto(h::EMU_PHYSICS_ME1, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.Write(tmp, 1);
      me->unlock();
    }
  
    if (getEMUHisto(h::EMU_PHYSICS_ME2, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.Write(tmp, 2);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_PHYSICS_ME3, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.Write(tmp, 3);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_PHYSICS_ME4, me)){
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.Write(tmp, 4);
      me->unlock();
    }

    if (getEMUHisto(h::EMU_PHYSICS_EMU, me)) {
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      summary.WriteMap(tmp);
      me->unlock();
    }

    // Looping via addresses (scope: side->station->ring) and
    // filling in HW efficiencies
    
    if (config->getPROCESS_EFF_PARAMETERS()) {

      Address adr;
      adr.mask.side = adr.mask.station = adr.mask.ring = true;
      adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  
      double e_detector = 0.0, e_side = 0.0, e_station = 0.0, e_ring = 0.0;
      
      const HistoId parameters [] = {
        h::PAR_CSC_SIDEMINUS_STATION01_RING01,
        h::PAR_CSC_SIDEMINUS_STATION01_RING02,
        h::PAR_CSC_SIDEMINUS_STATION01_RING03,
        h::PAR_CSC_SIDEMINUS_STATION01,
        h::PAR_CSC_SIDEMINUS_STATION02_RING01,
        h::PAR_CSC_SIDEMINUS_STATION02_RING02,
        h::PAR_CSC_SIDEMINUS_STATION02,
        h::PAR_CSC_SIDEMINUS_STATION03_RING01,
        h::PAR_CSC_SIDEMINUS_STATION03_RING02,
        h::PAR_CSC_SIDEMINUS_STATION03,
        h::PAR_CSC_SIDEMINUS_STATION04,
        h::PAR_CSC_SIDEMINUS,
        h::PAR_CSC_SIDEPLUS_STATION01_RING01,
        h::PAR_CSC_SIDEPLUS_STATION01_RING02,
        h::PAR_CSC_SIDEPLUS_STATION01_RING03,
        h::PAR_CSC_SIDEPLUS_STATION01,
        h::PAR_CSC_SIDEPLUS_STATION02_RING01,
        h::PAR_CSC_SIDEPLUS_STATION02_RING02,
        h::PAR_CSC_SIDEPLUS_STATION02,
        h::PAR_CSC_SIDEPLUS_STATION03_RING01,
        h::PAR_CSC_SIDEPLUS_STATION03_RING02,
        h::PAR_CSC_SIDEPLUS_STATION03,
        h::PAR_CSC_SIDEPLUS_STATION04,
        h::PAR_CSC_SIDEPLUS
      };

      unsigned int parameter = 0;

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
              if (getParHisto(parameters[parameter], me)) me->Fill(e_ring);
              parameter++;
            }
          }
          adr.mask.ring = false;
          e_station = e_station / summary.getDetector().NumberOfRings(adr.station);
          if (getParHisto(parameters[parameter], me)) me->Fill(e_station);
          parameter++;
          e_side += e_station;
        }
        adr.mask.station = false;
        e_side = e_side / N_STATIONS;
        if (getParHisto(parameters[parameter], me)) me->Fill(e_side);
        parameter++;
        e_detector += e_side; 
      }
      e_detector = e_detector / N_SIDES;
      if (getParHisto(h::PAR_REPORT_SUMMARY, me)) me->Fill(e_detector);

    }

  }

}

