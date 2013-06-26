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

#include "CSCDQM_EventProcessor.h"

namespace cscdqm {
  
  /**
   * @brief  Update Efficiency MOs
   */
  void EventProcessor::updateEfficiencyHistos() {

    MonitorObject *me = 0, *me1 = 0;

    if (config->getNEvents() > 0) {

      if (getEMUHisto(h::EMU_CSC_REPORTING, me)) {
  
        const TH2* rep = dynamic_cast<const TH2*>(me->getTH1());
  
        /**  Get CSC Reporting reference histogram */
        const TObject *tobj = me->getRefRootObject();
         
        /** If reference for CSC_Reporting is defined - use it
         *  Else - do it flat way */
         
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

      writeShifterHistograms();
  
      /**
       * Write summary information
       */
  
      if (getEMUHisto(h::EMU_PHYSICS_ME1, me)){
        LockType lock(me->mutex);
        TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
        summary.Write(tmp, 1);
      }
    
      if (getEMUHisto(h::EMU_PHYSICS_ME2, me)){
        LockType lock(me->mutex);
        TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
        summary.Write(tmp, 2);
      }
  
      if (getEMUHisto(h::EMU_PHYSICS_ME3, me)){
        LockType lock(me->mutex);
        TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
        summary.Write(tmp, 3);
      }
  
      if (getEMUHisto(h::EMU_PHYSICS_ME4, me)){
        LockType lock(me->mutex);
        TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
        summary.Write(tmp, 4);
      }
  
      if (getEMUHisto(h::EMU_PHYSICS_EMU, me)) {
        LockType lock(me->mutex);
        TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
        summary.WriteMap(tmp);
      }

    }

    /** Looping via addresses (scope: side->station->ring) and
     *  filling in HW efficiencies
     */
    
    if (config->getPROCESS_EFF_PARAMETERS()) {

      { // Compute DQM information parameters

        Address adr;
        adr.mask.side = adr.mask.station = adr.mask.ring = true;
        adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  
        double   e_detector = 0.0, e_side = 0.0, e_station = 0.0, e_ring = 0.0;
        uint32_t e_detector_ch = 0, e_side_ch = 0, e_station_ch = 0;
      
        const HistoId parameters [] = {
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
          h::PAR_CSC_SIDEPLUS_STATION04_RING01,
          h::PAR_CSC_SIDEPLUS_STATION04_RING02,
          h::PAR_CSC_SIDEPLUS_STATION04,
          h::PAR_CSC_SIDEPLUS,
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
          h::PAR_CSC_SIDEMINUS_STATION04_RING01,
          h::PAR_CSC_SIDEMINUS_STATION04_RING02,
          h::PAR_CSC_SIDEMINUS_STATION04,
          h::PAR_CSC_SIDEMINUS
        };

        bool calc = (config->getNEvents() > 0);

        if (!calc) {
          e_detector = e_side = e_station = e_ring = -1.0;
        }

        unsigned int parameter = 0;
        for (adr.side = 1; adr.side <= N_SIDES; adr.side++) {
          
          if (calc) {
            e_side = 0.0;
            e_side_ch = 0;
          }

          adr.mask.station = true;
          for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {
            
            if (calc) {
              e_station = 0.0;
              e_station_ch = 0;
            }
            
            adr.mask.ring = true;
            for (adr.ring = 1; adr.ring <= summary.getDetector().NumberOfRings(adr.station); adr.ring++) {

              if (calc) {
                e_ring = summary.GetEfficiencyHW(adr);
                uint32_t ch = summary.getDetector().NumberOfChambers(adr.station, adr.ring);
                e_station += (e_ring * ch);
                e_station_ch += ch;
              }

              if (summary.getDetector().NumberOfRings(adr.station) > 1) {
                if (getParHisto(parameters[parameter++], me)) me->Fill(e_ring);
              }

            }

            adr.mask.ring = false;
            if (calc) {
              e_side += e_station;
              e_side_ch += e_station_ch;
              e_station = e_station / e_station_ch;
            }

            if (getParHisto(parameters[parameter++], me)) me->Fill(e_station);

          }

          adr.mask.station = false;
          if (calc) {
            e_detector += e_side; 
            e_detector_ch += e_side_ch;
            e_side = e_side / e_side_ch;
          }

          if (getParHisto(parameters[parameter++], me)) me->Fill(e_side);

        }

        if (calc) {
          e_detector = e_detector / e_detector_ch;
        }

        if (getParHisto(h::PAR_REPORT_SUMMARY, me)) me->Fill(e_detector);

      }

    }

  }

  void EventProcessor::writeShifterHistograms() {

    MonitorObject *me = 0;

    //const int COLOR_WHITE   = 0;
    const int COLOR_GREEN   = 1;
    const int COLOR_RED     = 2;
    const int COLOR_BLUE    = 3;
    const int COLOR_GREY    = 4;
    const int COLOR_STANDBY = 5;

    if (getEMUHisto(h::EMU_CSC_STATS_SUMMARY, me)) {
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x1, COLOR_GREEN, true, false);
        summary.WriteChamberState(tmp, HWSTATUSERRORBITS, COLOR_RED, false, true);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_OCCUPANCY, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x4, COLOR_RED, true, false);
        summary.WriteChamberState(tmp, 0x8, COLOR_BLUE, false, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_FORMAT_ERR, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x10, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_L1SYNC_ERR, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x20, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_FIFOFULL_ERR, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x40, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_INPUTTO_ERR, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x80, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_WO_ALCT, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x100, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_WO_CLCT, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x200, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_WO_CFEB, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x400, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }

    if (getEMUHisto(h::EMU_CSC_STATS_CFEB_BWORDS, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());
      if (!config->getIN_FULL_STANDBY()) {
        summary.WriteChamberState(tmp, 0x800, COLOR_RED, true, false);
      }
      summary.WriteChamberState(tmp, 0x1000, COLOR_STANDBY, false);
      summary.WriteChamberState(tmp, 0x2, COLOR_GREY, false, false);
    }
    
  }

  /**
   * @brief  apply standby flags/parameters
   * @param standby standby flags
   */
  void EventProcessor::standbyEfficiencyHistos(HWStandbyType& standby) {

    Address adr;
    adr.mask.side = true;
    adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;

    adr.side = 1;
    summary.SetValue(adr, STANDBY, (standby.MeP ? 1 : 0));
    if (!standby.MeP) {
      summary.SetValue(adr, WAS_ON);
    }

    adr.side = 2;
    summary.SetValue(adr, STANDBY, (standby.MeM ? 1 : 0));
    if (!standby.MeM) {
      summary.SetValue(adr, WAS_ON);
    }

    MonitorObject *me = 0;
    if (getEMUHisto(h::EMU_CSC_STANDBY, me)){
      LockType lock(me->mutex);
      TH2* tmp = dynamic_cast<TH2*>(me->getTH1Lock());

      // All standby
      summary.WriteChamberState(tmp, 0x1000, 5);
       
      // Temporary in standby (was ON)
      summary.WriteChamberState(tmp, 0x3000, 1, false);

    }

  }

}

