/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor_processCSC.cc
 *
 *    Description:  Process Chamber
 *
 *        Version:  1.0
 *        Created:  10/03/2008 11:58:11 PM
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

  template <typename T>
  inline CSCCFEBDataWord const* const timeSample(T const& data, int nCFEB, int nSample, int nLayer, int nStrip) {
    return data.cfebData(nCFEB)->timeSlice(nSample)->timeSample(nLayer, nStrip, data.cfebData(nCFEB)->isDCFEB());
  }

  template <typename T>
  inline CSCCFEBTimeSlice const* const timeSlice(T const& data, int nCFEB, int nSample) {
    return data.cfebData(nCFEB)->timeSlice(nSample);
  }

  /**
   * @brief  Set a single bit in the 3D Histogram (aka EMU level event display). Checks if mo and x != null.
   * @param  eventNumber number of event
   * @param  mo Histogram
   * @param  x X bin number
   * @param  y Y bin number
   * @param  bit bit number to set
   */
  void EventProcessor::setEmuEventDisplayBit(MonitorObject*& mo,
                                             const unsigned int x,
                                             const unsigned int y,
                                             const unsigned int bit) {
    if (mo && x) {
      resetEmuEventDisplays();
      int bitset = (int)mo->GetBinContent(x, y);
      bitset |= 1 << bit;
      mo->SetBinContent(x, y, bitset);
    }
  }

  /**
   * @brief  Reset Emu level EventDisplay histograms once per event
   */
  void EventProcessor::resetEmuEventDisplays() {
    if (!EmuEventDisplayWasReset) {
      // Reseting EMU level Event displays
      MonitorObject* mo = nullptr;
      if (getEMUHisto(h::EMU_EVENT_DISPLAY_ANODE, mo)) {
        mo->getTH1Lock()->Reset("");
      }

      if (getEMUHisto(h::EMU_EVENT_DISPLAY_CATHODE, mo)) {
        mo->getTH1Lock()->Reset("");
      }

      if (getEMUHisto(h::EMU_EVENT_DISPLAY_XY, mo)) {
        mo->getTH1Lock()->Reset("");
      }

      EmuEventDisplayWasReset = true;
    }
  }

  /**
   * @brief  Process Chamber Data and fill MOs
   * @param  data Chamber data to process
   * @param dduID DDU identifier
   */
  void EventProcessor::processCSC(const CSCEventData& data, const int dduID, const CSCDCCExaminer& binChecker) {
    config->incNUnpackedCSC();

    int FEBunpacked = 0;
    int alct_unpacked = 0;
    int tmb_unpacked = 0;
    int cfeb_unpacked = 0;

    int alct_keywg = -1;
    int clct_kewdistrip = -1;

    bool L1A_out_of_sync = false;

    int nCFEBs = 5;

    MonitorObject* mo = nullptr;

    /**   DMB Found */
    /**   Unpacking of DMB Header and trailer */
    const CSCDMBHeader* dmbHeader = data.dmbHeader();
    const CSCDMBTrailer* dmbTrailer = data.dmbTrailer();
    if (!dmbHeader && !dmbTrailer) {
      LOG_ERROR << "Can not unpack DMB Header or/and Trailer";
      return;
    }

    theFormatVersion = data.getFormatVersion();

    /**  Unpacking of Chamber Identification number */
    unsigned int crateID = 0xFF;
    unsigned int dmbID = 0xF;
    unsigned int chamberID = 0xFFF;

    crateID = dmbHeader->crateID();
    dmbID = dmbHeader->dmbID();
    chamberID = (((crateID) << 4) + dmbID) & 0xFFF;

    const std::string cscTag = CSCHistoDef::getPath(crateID, dmbID);

    unsigned long errors = binChecker.errorsForChamber(chamberID);
    if ((errors & config->getBINCHECK_MASK()) > 0) {
      LOG_WARN << "Format Errors " << cscTag << ": 0x" << std::hex << errors << " Skipped CSC Unpacking";
      return;
    }

    unsigned int cscType = 0;
    unsigned int cscPosition = 0;
    if (!getCSCFromMap(crateID, dmbID, cscType, cscPosition))
      return;

    CSCDetId cid;
    if (!config->fnGetCSCDetId(crateID, dmbID, cid)) {
      return;
    }

    // Check if ME11 with PostLS1 readout (7 DCFEBs)
    if ((cscType == 8 || cscType == 9) && theFormatVersion >= 2013)
      nCFEBs = 7;

    // Check if in standby!
    if (summary.isChamberStandby(cid)) {
      return;
    }

    double DMBEvents = 0.0;
    DMBEvents = config->getChamberCounterValue(DMB_EVENTS, crateID, dmbID);

    // Get Event display plot number and next plot object
    uint32_t evDisplNo = config->getChamberCounterValue(EVENT_DISPLAY_PLOT, crateID, dmbID);
    evDisplNo += 1;
    if (evDisplNo >= 5) {
      config->setChamberCounterValue(EVENT_DISPLAY_PLOT, crateID, dmbID, 0);
    } else {
      config->setChamberCounterValue(EVENT_DISPLAY_PLOT, crateID, dmbID, evDisplNo);
    }
    MonitorObject* mo_EventDisplay = nullptr;
    if (getCSCHisto(h::CSC_EVENT_DISPLAY_NOXX, crateID, dmbID, evDisplNo, mo_EventDisplay)) {
      mo_EventDisplay->getTH1Lock()->Reset("");
    }

    // Receiving EMU Event displays
    MonitorObject *mo_Emu_EventDisplay_Anode = nullptr, *mo_Emu_EventDisplay_Cathode = nullptr,
                  *mo_Emu_EventDisplay_XY = nullptr;
    getEMUHisto(h::EMU_EVENT_DISPLAY_ANODE, mo_Emu_EventDisplay_Anode);
    getEMUHisto(h::EMU_EVENT_DISPLAY_CATHODE, mo_Emu_EventDisplay_Cathode);
    getEMUHisto(h::EMU_EVENT_DISPLAY_XY, mo_Emu_EventDisplay_XY);

    // Global chamber index
    uint32_t glChamberIndex = 0;

    if (mo_EventDisplay) {
      mo_EventDisplay->SetBinContent(1, 1, cid.endcap());
      mo_EventDisplay->SetBinContent(1, 2, cid.station());
      mo_EventDisplay->SetBinContent(1, 3, cid.ring());
      mo_EventDisplay->SetBinContent(1, 4, cscPosition);
      mo_EventDisplay->SetBinContent(1, 5, crateID);
      mo_EventDisplay->SetBinContent(1, 6, dmbID);
      mo_EventDisplay->SetBinContent(1, 7, dmbHeader->l1a24());
      if (mo_Emu_EventDisplay_Anode || mo_Emu_EventDisplay_Cathode || mo_Emu_EventDisplay_XY) {
        glChamberIndex = summary.getDetector().GlobalChamberIndex(cid.endcap(), cid.station(), cid.ring(), cscPosition);
      }
    }

    config->copyChamberCounterValue(DMB_EVENTS, DMB_TRIGGERS, crateID, dmbID);

    if (cscPosition && getEMUHisto(h::EMU_CSC_UNPACKED, mo)) {
      mo->Fill(cscPosition, cscType);
    }

    /**     Efficiency of the chamber */
    float DMBEff = float(DMBEvents) / float(config->getNEvents());
    if (DMBEff > 1.0) {
      LOG_ERROR << cscTag << " has efficiency " << DMBEff << " which is greater than 1";
    }

    /**  Unpacking L1A number from DMB header */
    /**  DMB L1A: 8bits (256) */
    /**  DDU L1A: 24bits */
    int dmbHeaderL1A = dmbHeader->l1a24() % 64;
    /**  Calculate difference between L1A numbers from DDU and DMB */
    int dmb_ddu_l1a_diff = (int)(dmbHeaderL1A - (int)(L1ANumber % 64));
    if (dmb_ddu_l1a_diff != 0)
      L1A_out_of_sync = true;

    /** LOG_DEBUG << dmbHeaderL1A << " : DMB L1A - DDU L1A = " << dmb_ddu_l1a_diff; */

    if (getCSCHisto(h::CSC_DMB_L1A_DISTRIB, crateID, dmbID, mo))
      mo->Fill(dmbHeaderL1A);

    if (getCSCHisto(h::CSC_DMB_DDU_L1A_DIFF, crateID, dmbID, mo)) {
      if (dmb_ddu_l1a_diff < -32) {
        mo->Fill(dmb_ddu_l1a_diff + 64);
      } else {
        if (dmb_ddu_l1a_diff >= 32)
          mo->Fill(dmb_ddu_l1a_diff - 64);
        else
          mo->Fill(dmb_ddu_l1a_diff);
      }
      mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
    }

    if (getCSCHisto(h::CSC_DMB_L1A_VS_DDU_L1A, crateID, dmbID, mo))
      mo->Fill((int)(L1ANumber & 0xFF), (int)dmbHeaderL1A);

    /**     Unpacking BXN number from DMB header */
    int dmbHeaderBXN = 0;
    int dmb_ddu_bxn_diff = 0;

    /**  == DMB BXN: 12bits (4096) call bxn12(), bxn() return 7bits value */
    /**  == DDU BXN: 12bits (4096)  */

    /**  == Use 6-bit BXN */
    dmbHeaderBXN = dmbHeader->bxn12();
    /**  Calculation difference between BXN numbers from DDU and DMB */

    /**   dmb_ddu_bxn_diff = (int)(dmbHeaderBXN-(int)(BXN&0x7F)); // For older DMB */
    dmb_ddu_bxn_diff = dmbHeaderBXN % 64 - BXN % 64;
    /** LOG_DEBUG << dmbHeaderBXN << " : DMB BXN - DDU BXN = " << dmb_ddu_bxn_diff; */
    if (getCSCHisto(h::CSC_DMB_BXN_DISTRIB, crateID, dmbID, mo))
      mo->Fill((int)(dmbHeader->bxn12()));

    if (getCSCHisto(h::CSC_DMB_DDU_BXN_DIFF, crateID, dmbID, mo)) {
      if (dmb_ddu_bxn_diff < -32)
        mo->Fill(dmb_ddu_bxn_diff + 64);
      else {
        if (dmb_ddu_bxn_diff >= 32)
          mo->Fill(dmb_ddu_bxn_diff - 64);
        else
          mo->Fill(dmb_ddu_bxn_diff);
      }
      mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
    }

    if (getCSCHisto(h::CSC_DMB_BXN_VS_DDU_BXN, crateID, dmbID, mo))
      mo->Fill(((int)(BXN)) % 256, ((int)dmbHeaderBXN) % 256);

    /**     Unpacking CFEB information from DMB header */
    int cfeb_dav = 0;
    int cfeb_dav_num = 0;
    int cfeb_movlp = 0;
    int dmb_cfeb_sync = 0;

    cfeb_dav = (int)dmbHeader->cfebAvailable();
    for (int i = 0; i < nCFEBs; i++)
      cfeb_dav_num += (cfeb_dav >> i) & 0x1;
    cfeb_movlp = (int)dmbHeader->cfebMovlp();
    dmb_cfeb_sync = (int)dmbHeader->dmbCfebSync();

    if (getCSCHisto(h::CSC_DMB_CFEB_DAV, crateID, dmbID, mo)) {
      for (int i = 0; i < nCFEBs; i++) {
        int cfeb_present = (cfeb_dav >> i) & 0x1;
        if (cfeb_present) {
          mo->Fill(i);
        }
      }
    }

    if (getCSCHisto(h::CSC_DMB_CFEB_DAV_MULTIPLICITY, crateID, dmbID, mo))
      mo->Fill(cfeb_dav_num);
    if (getCSCHisto(h::CSC_DMB_CFEB_MOVLP, crateID, dmbID, mo))
      mo->Fill(cfeb_movlp);
    if (getCSCHisto(h::CSC_DMB_CFEB_SYNC, crateID, dmbID, mo))
      mo->Fill(dmb_cfeb_sync);

    if (getEMUHisto(h::EMU_DMB_UNPACKED, mo)) {
      mo->Fill(crateID, dmbID);
      /**   mo->SetEntries(config->getNEvents()); */
    }

    if (getCSCHisto(h::CSC_DMB_CFEB_ACTIVE, crateID, dmbID, mo))
      mo->Fill(dmbHeader->cfebActive());  //KK

    if (getCSCHisto(h::CSC_DMB_L1_PIPE, crateID, dmbID, mo))
      mo->Fill(dmbTrailer->dmb_l1pipe());

    /**  DMB input (7 in total) FIFO stuff goes here */
    if (getCSCHisto(h::CSC_DMB_FIFO_STATS, crateID, dmbID, mo)) {
      if (dmbTrailer->tmb_empty() == 1)
        mo->Fill(1.0, 0.0);  //KK
      if (dmbTrailer->tmb_half() == 0)
        mo->Fill(1.0, 1.0);
      if (dmbTrailer->tmb_full() == 1)
        mo->Fill(1.0, 2.0);  //KK
      if (dmbTrailer->alct_empty() == 1)
        mo->Fill(0.0, 0.0);
      if (dmbTrailer->alct_half() == 0)
        mo->Fill(0.0, 1.0);
      if (dmbTrailer->alct_full() == 1)
        mo->Fill(0.0, 2.0);  //KK 0->1
      for (int i = 0; i < nCFEBs; i++) {
        if ((int)((dmbTrailer->cfeb_empty() >> i) & 0x1) == 1)
          mo->Fill(i + 2, 0.0);
        if ((int)((dmbTrailer->cfeb_half() >> i) & 0x1) == 0)
          mo->Fill(i + 2, 1);
        if ((int)((dmbTrailer->cfeb_full() >> i) & 0x1) == 1) {
          mo->Fill(i + 2, 2);
        }
      }
      mo->SetEntries((int)DMBEvents);
    }

    /**  DMB input timeout (total 15 bits) goes here */
    if (getCSCHisto(h::CSC_DMB_FEB_TIMEOUTS, crateID, dmbID, mo)) {
      if ((dmbTrailer->tmb_starttimeout() == 0) && (dmbTrailer->alct_starttimeout() == 0) &&
          (dmbTrailer->cfeb_starttimeout() == 0) && (dmbTrailer->cfeb_endtimeout() == 0)) {
        mo->Fill(0.0);
      } else {
        if (dmbTrailer->alct_starttimeout())
          mo->Fill(1);
        if (dmbTrailer->tmb_starttimeout())
          mo->Fill(2);
        if (dmbTrailer->alct_endtimeout())
          mo->Fill(8);  // KK
        if (dmbTrailer->tmb_endtimeout())
          mo->Fill(9);  // KK
      }
      for (int i = 0; i < nCFEBs; i++) {
        if ((dmbTrailer->cfeb_starttimeout() >> i) & 0x1) {
          mo->Fill(i + 3);
        }
        if ((dmbTrailer->cfeb_endtimeout() >> i) & 0x1) {
          mo->Fill(i + 10);  // KK 8->10
        }
      }
      mo->SetEntries((int)DMBEvents);
    }

    /**       Get FEBs Data Available Info */
    int alct_dav = dmbHeader->nalct();
    int tmb_dav = dmbHeader->nclct();
    int cfeb_dav2 = 0;
    for (int i = 0; i < nCFEBs; i++)
      cfeb_dav2 = cfeb_dav2 + (int)((dmbHeader->cfebAvailable() >> i) & 0x1);

    /**       Fill Hisogram for FEB DAV Efficiency */

    if ((alct_dav > 0) && (getCSCHisto(h::CSC_DMB_FEB_DAV_RATE, crateID, dmbID, mo))) {
      mo->Fill(0.0);
      float alct_dav_number = mo->GetBinContent(1);
      if (getCSCHisto(h::CSC_DMB_FEB_DAV_EFFICIENCY, crateID, dmbID, mo)) {
        mo->SetBinContent(1, ((float)alct_dav_number / (float)(DMBEvents)*100.0));
        mo->SetEntries((int)DMBEvents);
      }
    }

    if ((tmb_dav > 0) && (getCSCHisto(h::CSC_DMB_FEB_DAV_RATE, crateID, dmbID, mo))) {
      mo->Fill(1.0);
      float tmb_dav_number = mo->GetBinContent(2);
      if (getCSCHisto(h::CSC_DMB_FEB_DAV_EFFICIENCY, crateID, dmbID, mo)) {
        mo->SetBinContent(2, ((float)tmb_dav_number / (float)(DMBEvents)*100.0));
        mo->SetEntries((int)DMBEvents);
      }
    }

    if ((cfeb_dav2 > 0) && (getCSCHisto(h::CSC_DMB_FEB_DAV_RATE, crateID, dmbID, mo))) {
      mo->Fill(2.0);
      float cfeb_dav2_number = mo->GetBinContent(3);
      if (getCSCHisto(h::CSC_DMB_FEB_DAV_EFFICIENCY, crateID, dmbID, mo)) {
        mo->SetBinContent(3, ((float)cfeb_dav2_number / (float)(DMBEvents)*100.0));
        mo->SetEntries((int)DMBEvents);
      }
    }

    float feb_combination_dav = -1.0;
    /**       Fill Hisogram for Different Combinations of FEB DAV Efficiency */
    if (getCSCHisto(h::CSC_DMB_FEB_COMBINATIONS_DAV_RATE, crateID, dmbID, mo)) {
      if (alct_dav == 0 && tmb_dav == 0 && cfeb_dav2 == 0)
        feb_combination_dav = 0.0;  // Nothing
      if (alct_dav > 0 && tmb_dav == 0 && cfeb_dav2 == 0)
        feb_combination_dav = 1.0;  // ALCT Only
      if (alct_dav == 0 && tmb_dav > 0 && cfeb_dav2 == 0)
        feb_combination_dav = 2.0;  // TMB Only
      if (alct_dav == 0 && tmb_dav == 0 && cfeb_dav2 > 0)
        feb_combination_dav = 3.0;  // CFEB Only
      if (alct_dav == 0 && tmb_dav > 0 && cfeb_dav2 > 0)
        feb_combination_dav = 4.0;  // TMB+CFEB
      if (alct_dav > 0 && tmb_dav > 0 && cfeb_dav2 == 0)
        feb_combination_dav = 5.0;  // ALCT+TMB
      if (alct_dav > 0 && tmb_dav == 0 && cfeb_dav2 > 0)
        feb_combination_dav = 6.0;  // ALCT+CFEB
      if (alct_dav > 0 && tmb_dav > 0 && cfeb_dav2 > 0)
        feb_combination_dav = 7.0;  // ALCT+TMB+CFEB
      mo->Fill(feb_combination_dav);
      float feb_combination_dav_number = mo->GetBinContent((int)(feb_combination_dav + 1.0));
      if (getCSCHisto(h::CSC_DMB_FEB_COMBINATIONS_DAV_EFFICIENCY, crateID, dmbID, mo)) {
        mo->SetBinContent((int)(feb_combination_dav + 1.0),
                          ((float)feb_combination_dav_number / (float)(DMBEvents)*100.0));
        mo->SetEntries((int)DMBEvents);
      }
    }

    /** ALCT Found */
    if (data.nalct()) {
      const CSCALCTHeader* alctHeader = data.alctHeader();
      int fwVersion = alctHeader->alctFirmwareVersion();
      int fwRevision = alctHeader->alctFirmwareRevision();
      const CSCALCTTrailer* alctTrailer = data.alctTrailer();
      const CSCAnodeData* alctData = data.alctData();

      if (alctHeader && alctTrailer) {
        /** Summary plot for chambers with detected Run3 ALCT firmware */
        if (getEMUHisto(h::EMU_CSC_RUN3_ALCT_FORMAT, mo)) {
          /// ALCT Run3 firmware revision should be > 5
          if (fwRevision > 5)
            mo->SetBinContent(cscPosition, cscType + 1, fwRevision);
        }
        std::vector<CSCALCTDigi> alctsDatasTmp = alctHeader->ALCTDigis();
        std::vector<CSCALCTDigi> alctsDatas;

        for (uint32_t lct = 0; lct < alctsDatasTmp.size(); lct++) {
          if (alctsDatasTmp[lct].isValid())
            alctsDatas.push_back(alctsDatasTmp[lct]);
        }

        FEBunpacked = FEBunpacked + 1;
        alct_unpacked = 1;

        /**  Set number of ALCT-events to third bin */
        if (getCSCHisto(h::CSC_CSC_RATE, crateID, dmbID, mo)) {
          mo->Fill(2);
          uint32_t ALCTEvent = (uint32_t)mo->GetBinContent(3);
          config->setChamberCounterValue(ALCT_TRIGGERS, crateID, dmbID, ALCTEvent);
          if (getCSCHisto(h::CSC_CSC_EFFICIENCY, crateID, dmbID, mo)) {
            if (config->getNEvents() > 0) {
              /** KK */
              /** h[hname]->SetBinContent(3, ((float)ALCTEvent/(float)(config->getNEvents()) * 100.0)); */
              mo->SetBinContent(1, ((float)ALCTEvent / (float)(DMBEvents)*100.0));
              /** KKend */
              mo->SetEntries((int)DMBEvents);
            }
          }
        }

        if ((alct_dav > 0) && (getCSCHisto(h::CSC_DMB_FEB_UNPACKED_VS_DAV, crateID, dmbID, mo))) {
          mo->Fill(0.0, 0.0);
        }

        /**  == ALCT2007 L1A: 12bits (4096) */
        /**  == ALCT2006 L1A: 4bits (16) */
        if (getCSCHisto(h::CSC_ALCT_L1A, crateID, dmbID, mo))
          mo->Fill((int)(alctHeader->L1Acc()));

        /**  == Use 6-bit L1A       */
        if (getCSCHisto(h::CSC_ALCT_DMB_L1A_DIFF, crateID, dmbID, mo)) {
          /**  int alct_dmb_l1a_diff = (int)((dmbHeader->l1a()&0xF)-alctHeader->L1Acc()); */
          int alct_dmb_l1a_diff = (int)(alctHeader->L1Acc() % 64 - dmbHeader->l1a24() % 64);
          if (alct_dmb_l1a_diff != 0)
            L1A_out_of_sync = true;
          if (alct_dmb_l1a_diff < -32)
            mo->Fill(alct_dmb_l1a_diff + 64);
          else {
            if (alct_dmb_l1a_diff >= 32)
              mo->Fill(alct_dmb_l1a_diff - 64);
            else
              mo->Fill(alct_dmb_l1a_diff);
          }
          mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
        }

        /**  if (getCSCHisto(h::CSC_DMB_L1A_VS_ALCT_L1A, crateID, dmbID, mo)) mo->Fill(alctHeader->L1Acc(),dmbHeader->l1a()); */
        if (getCSCHisto(h::CSC_DMB_L1A_VS_ALCT_L1A, crateID, dmbID, mo))
          mo->Fill(alctHeader->L1Acc() % 256, dmbHeader->l1a24() % 256);

        /**  === ALCT BXN: 12bits (4096) */
        /**  === Use 6-bit BXN */
        if (getCSCHisto(h::CSC_ALCT_DMB_BXN_DIFF, crateID, dmbID, mo)) {
          int alct_dmb_bxn_diff = (int)(alctHeader->BXNCount() - dmbHeader->bxn12());
          if (alct_dmb_bxn_diff > 0)
            alct_dmb_bxn_diff -= 3564;
          alct_dmb_bxn_diff %= 32;
          mo->Fill(alct_dmb_bxn_diff);
          mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
        }

        if (getCSCHisto(h::CSC_ALCT_BXN, crateID, dmbID, mo))
          mo->Fill(alctHeader->BXNCount());

        /**  if (getCSCHisto(h::CSC_ALCT_BXN_VS_DMB_BXN, crateID, dmbID, mo)) mo->Fill((int)((alctHeader->BXNCount())), (int)(dmbHeader->bxn())); */
        if (getCSCHisto(h::CSC_ALCT_BXN_VS_DMB_BXN, crateID, dmbID, mo))
          mo->Fill((int)((alctHeader->BXNCount()) % 256), (int)(dmbHeader->bxn12()) % 256);

        if (getCSCHisto(h::CSC_ALCT_NUMBER_RATE, crateID, dmbID, mo)) {
          mo->Fill(alctsDatas.size());
          int nALCT = (int)mo->GetBinContent((int)(alctsDatas.size() + 1));
          if (getCSCHisto(h::CSC_ALCT_NUMBER_EFFICIENCY, crateID, dmbID, mo))
            mo->SetBinContent((int)(alctsDatas.size() + 1), (float)(nALCT) / (float)(DMBEvents)*100.0);
        }

        if (getCSCHisto(h::CSC_ALCT_WORD_COUNT, crateID, dmbID, mo))
          mo->Fill((int)(alctTrailer->wordCount()));

        /** LOG_DEBUG <<  "ALCT Trailer Word Count = " << dec << (int)alctTrailer->wordCount(); */

        if (alctsDatas.size() == 2) {
          if (getCSCHisto(h::CSC_ALCT1_VS_ALCT0_KEYWG, crateID, dmbID, mo))
            mo->Fill(alctsDatas[0].getKeyWG(), alctsDatas[1].getKeyWG());
        }

        /** Run3 ALCT HMT bits from ALCT data */
        std::vector<CSCShowerDigi> alctShowers = alctHeader->alctShowerDigis();
        for (unsigned bx = 0; bx < alctShowers.size(); bx++) {
          if (getCSCHisto(h::CSC_RUN3_ALCT_HMT_BITS_VS_BX, crateID, dmbID, mo)) {
            mo->Fill(alctShowers[bx].bitsInTime(), bx);
          }
          /** Summary occupancy plot for ANode HMT bits per BX from ALCT */
          if (getEMUHisto(h::EMU_CSC_ANODE_HMT_ALCT_REPORTING, mo)) {
            if (alctShowers[bx].isValid())
              mo->Fill(cscPosition, cscType);
          }
        }

        MonitorObject* mo_CSC_ALCT0_BXN_mean = nullptr;
        getEMUHisto(h::EMU_CSC_ALCT0_BXN_MEAN, mo_CSC_ALCT0_BXN_mean);

        MonitorObject* mo_CSC_ALCT0_BXN_rms = nullptr;
        getEMUHisto(h::EMU_CSC_ALCT0_BXN_RMS, mo_CSC_ALCT0_BXN_rms);

        MonitorObject* mo_CSC_Plus_endcap_ALCT0_dTime = nullptr;
        getEMUHisto(h::EMU_CSC_ALCT0_ENDCAP_PLUS_DTIME, mo_CSC_Plus_endcap_ALCT0_dTime);

        MonitorObject* mo_CSC_Minus_endcap_ALCT0_dTime = nullptr;
        getEMUHisto(h::EMU_CSC_ALCT0_ENDCAP_MINUS_DTIME, mo_CSC_Minus_endcap_ALCT0_dTime);

        for (uint32_t lct = 0; lct < alctsDatas.size(); lct++) {
          /**  TODO: Add support for more than 2 ALCTs */
          if (lct >= 2)
            continue;

          if (getCSCHisto(h::CSC_ALCTXX_KEYWG, crateID, dmbID, lct, mo)) {
            mo->Fill(alctsDatas[lct].getKeyWG());
          }

          if (lct == 0)
            alct_keywg = alctsDatas[lct].getKeyWG();

          int alct_dtime = 0;
          if (fwVersion == 2007) {
            alct_dtime = alctsDatas[lct].getBX();
          } else {
            // Older 2006 Format
            alct_dtime = (int)(alctsDatas[lct].getBX() - (alctHeader->BXNCount() & 0x1F));
          }

          // == Those two summary histos need to be outside of per-chamber CSC_ALCTXX_DTIME histo check.
          //    Otherwise will be empty in Offline DQM
          if (lct == 0) {
            if (cid.endcap() == 1) {
              if (mo_CSC_Plus_endcap_ALCT0_dTime)
                mo_CSC_Plus_endcap_ALCT0_dTime->Fill(alct_dtime);
            }
            if (cid.endcap() == 2) {
              if (mo_CSC_Minus_endcap_ALCT0_dTime)
                mo_CSC_Minus_endcap_ALCT0_dTime->Fill(alct_dtime);
            }
          }

          if (getCSCHisto(h::CSC_ALCTXX_DTIME, crateID, dmbID, lct, mo)) {
            if (alct_dtime < -16) {
              mo->Fill(alct_dtime + 32);
            } else {
              if (alct_dtime >= 16)
                mo->Fill(alct_dtime - 32);
              else
                mo->Fill(alct_dtime);
            }

            mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");

            double dTime_mean = mo->getTH1()->GetMean();
            double dTime_rms = mo->getTH1()->GetRMS();

            // == For ALCT0 Fill Summary dTime Histograms
            if (lct == 0) {
              if (cscPosition && mo_CSC_ALCT0_BXN_mean) {
                mo_CSC_ALCT0_BXN_mean->SetBinContent(cscPosition, cscType + 1, dTime_mean);
              }
              if (cscPosition && mo_CSC_ALCT0_BXN_rms) {
                mo_CSC_ALCT0_BXN_rms->SetBinContent(cscPosition, cscType + 1, dTime_rms);
              }
            }
          }

          if (getCSCHisto(h::CSC_ALCTXX_DTIME_VS_KEYWG, crateID, dmbID, lct, mo)) {
            if (alct_dtime < -16) {
              mo->Fill(alctsDatas[lct].getKeyWG(), alct_dtime + 32);
            } else {
              if (alct_dtime >= 16)
                mo->Fill(alctsDatas[lct].getKeyWG(), alct_dtime - 32);
              else
                mo->Fill(alctsDatas[lct].getKeyWG(), alct_dtime);
            }
          }

          if (getCSCHisto(h::CSC_ALCTXX_DTIME_PROFILE, crateID, dmbID, lct, mo)) {
            if (alct_dtime < -16) {
              mo->Fill(alctsDatas[lct].getKeyWG(), alct_dtime + 32);
            } else {
              if (alct_dtime >= 16)
                mo->Fill(alctsDatas[lct].getKeyWG(), alct_dtime - 32);
              else
                mo->Fill(alctsDatas[lct].getKeyWG(), alct_dtime);
            }
          }

          int alct_bxn = alctsDatas[lct].getBX();
          if (fwVersion == 2007) {
            alct_bxn = (alct_bxn + alctHeader->BXNCount()) & 0x1F;
          }

          if (getCSCHisto(h::CSC_ALCTXX_BXN, crateID, dmbID, lct, mo))
            mo->Fill(alct_bxn);

          if (getCSCHisto(h::CSC_ALCTXX_QUALITY, crateID, dmbID, lct, mo))
            mo->Fill(alctsDatas[lct].getKeyWG(), alctsDatas[lct].getQuality());

          if (mo_EventDisplay) {
            mo_EventDisplay->SetBinContent(2, alctsDatas[lct].getKeyWG(), alct_bxn + 1);
            mo_EventDisplay->SetBinContent(3, alctsDatas[lct].getKeyWG(), alctsDatas[lct].getQuality());
          }

          if (getCSCHisto(h::CSC_ALCTXX_QUALITY_DISTR, crateID, dmbID, lct, mo)) {
            mo->Fill(alctsDatas[lct].getQuality());
            if (lct == 0) {
              MonitorObject* mo1 = nullptr;
              if (cscPosition && getEMUHisto(h::EMU_CSC_ALCT0_QUALITY, mo1)) {
                mo1->SetBinContent(cscPosition, cscType + 1, mo->getTH1()->GetMean());
              }
            }
          }

          if (getCSCHisto(h::CSC_ALCTXX_QUALITY_PROFILE, crateID, dmbID, lct, mo))
            mo->Fill(alctsDatas[lct].getKeyWG(), alctsDatas[lct].getQuality());

          if (getCSCHisto(h::CSC_ALCTXX_PATTERN, crateID, dmbID, lct, mo)) {
            int pattern = (alctsDatas[lct].getAccelerator() << 1) + alctsDatas[lct].getCollisionB();
            int keywg = alctsDatas[lct].getKeyWG();
            mo->Fill(keywg, pattern);
          }

          if (getCSCHisto(h::CSC_ALCTXX_PATTERN_DISTR, crateID, dmbID, lct, mo)) {
            int pattern = (alctsDatas[lct].getAccelerator() << 1) + alctsDatas[lct].getCollisionB();
            mo->Fill(pattern);
          }
        }

        int NumberOfLayersWithHitsInALCT = 0;
        int NumberOfWireGroupsWithHitsInALCT = 0;

        if (alctData) {
          MonitorObject* mo_AFEB_RawHits_TimeBins = nullptr;
          getCSCHisto(h::CSC_CFEB_AFEB_RAWHITS_TIMEBINS, crateID, dmbID, mo_AFEB_RawHits_TimeBins);

          MonitorObject* mo_CSC_Plus_endcap_AFEB_RawHits_Time = nullptr;
          getEMUHisto(h::EMU_CSC_AFEB_ENDCAP_PLUS_RAWHITS_TIME, mo_CSC_Plus_endcap_AFEB_RawHits_Time);

          MonitorObject* mo_CSC_Minus_endcap_AFEB_RawHits_Time = nullptr;
          getEMUHisto(h::EMU_CSC_AFEB_ENDCAP_MINUS_RAWHITS_TIME, mo_CSC_Minus_endcap_AFEB_RawHits_Time);

          MonitorObject* mo_CSC_AFEB_RawHits_Time_mean = nullptr;
          getEMUHisto(h::EMU_CSC_AFEB_RAWHITS_TIME_MEAN, mo_CSC_AFEB_RawHits_Time_mean);

          MonitorObject* mo_CSC_AFEB_RawHits_Time_rms = nullptr;
          getEMUHisto(h::EMU_CSC_AFEB_RAWHITS_TIME_RMS, mo_CSC_AFEB_RawHits_Time_rms);

          for (int nLayer = 1; nLayer <= 6; nLayer++) {
            int wg_previous = -1;
            int tbin_previous = -1;
            bool CheckLayerALCT = true;

            std::vector<CSCWireDigi> wireDigis = alctData->wireDigis(nLayer);
            for (std::vector<CSCWireDigi>::iterator wireDigisItr = wireDigis.begin(); wireDigisItr != wireDigis.end();
                 ++wireDigisItr) {
              int wg = wireDigisItr->getWireGroup();
              std::vector<int> tbins = wireDigisItr->getTimeBinsOn();
              int tbin = wireDigisItr->getTimeBin();

              if (mo_EventDisplay) {
                mo_EventDisplay->SetBinContent(nLayer + 3, wg - 1, tbin + 1);
                setEmuEventDisplayBit(mo_Emu_EventDisplay_Anode, glChamberIndex, wg - 1, nLayer - 1);
                setEmuEventDisplayBit(mo_Emu_EventDisplay_XY, glChamberIndex, wg - 1, nLayer - 1);
              }

              if (CheckLayerALCT) {
                NumberOfLayersWithHitsInALCT = NumberOfLayersWithHitsInALCT + 1;
                CheckLayerALCT = false;
              }

              for (uint32_t n = 0; n < tbins.size(); n++) {
                tbin = tbins[n];
                if (wg != wg_previous || (tbin != tbin_previous + 1 && tbin != tbin_previous - 1)) {
                  if (getCSCHisto(h::CSC_ALCTTIME_LYXX, crateID, dmbID, nLayer, mo))
                    mo->Fill(wg - 1, tbin);

                  if (getCSCHisto(h::CSC_ALCTTIME_LYXX_PROFILE, crateID, dmbID, nLayer, mo))
                    mo->Fill(wg - 1, tbin);

                  if (mo_AFEB_RawHits_TimeBins)
                    mo_AFEB_RawHits_TimeBins->Fill(tbin);

                  if (cid.endcap() == 1) {
                    if (mo_CSC_Plus_endcap_AFEB_RawHits_Time)
                      mo_CSC_Plus_endcap_AFEB_RawHits_Time->Fill(tbin);
                  }
                  if (cid.endcap() == 2) {
                    if (mo_CSC_Minus_endcap_AFEB_RawHits_Time)
                      mo_CSC_Minus_endcap_AFEB_RawHits_Time->Fill(tbin);
                  }

                  if (getCSCHisto(h::CSC_ALCT_LYXX_RATE, crateID, dmbID, nLayer, mo)) {
                    mo->Fill(wg - 1);
                    int number_wg = (int)(mo->GetBinContent(wg));
                    Double_t Number_of_entries_ALCT = mo->GetEntries();
                    if (getCSCHisto(h::CSC_ALCT_LYXX_EFFICIENCY, crateID, dmbID, nLayer, mo)) {
                      mo->SetBinContent(wg, ((float)number_wg));
                      if ((Double_t)(DMBEvents) > 0.0) {
                        mo->SetNormFactor(100.0 * Number_of_entries_ALCT / (Double_t)(DMBEvents));
                      } else {
                        mo->SetNormFactor(100.0);
                      }
                      mo->SetEntries((int)DMBEvents);
                    }
                  }
                }
                if (wg != wg_previous) {
                  NumberOfWireGroupsWithHitsInALCT = NumberOfWireGroupsWithHitsInALCT + 1;
                }

                wg_previous = wg;
                tbin_previous = tbin;
              }
            }

            // Fill Summary Anode Raw Hits Timing Plots
            if (mo_AFEB_RawHits_TimeBins) {
              double rawhits_time_mean = mo_AFEB_RawHits_TimeBins->getTH1()->GetMean();
              double rawhits_time_rms = mo_AFEB_RawHits_TimeBins->getTH1()->GetRMS();

              if (cscPosition && mo_CSC_AFEB_RawHits_Time_mean) {
                mo_CSC_AFEB_RawHits_Time_mean->SetBinContent(cscPosition, cscType + 1, rawhits_time_mean);
              }

              if (cscPosition && mo_CSC_AFEB_RawHits_Time_rms) {
                mo_CSC_AFEB_RawHits_Time_rms->SetBinContent(cscPosition, cscType + 1, rawhits_time_rms);
              }
            }
          }

        } else {
          LOG_ERROR << cscTag << " Can not unpack Anode Data";
        }

        if (getCSCHisto(h::CSC_ALCT_NUMBER_OF_LAYERS_WITH_HITS, crateID, dmbID, mo)) {
          mo->Fill(NumberOfLayersWithHitsInALCT);
          MonitorObject* mo1 = nullptr;
          if (cscPosition && getEMUHisto(h::EMU_CSC_ALCT_PLANES_WITH_HITS, mo1)) {
            mo1->SetBinContent(cscPosition, cscType + 1, mo->getTH1()->GetMean());
          }
        }

        if (getCSCHisto(h::CSC_ALCT_NUMBER_OF_WIREGROUPS_WITH_HITS, crateID, dmbID, mo))
          mo->Fill(NumberOfWireGroupsWithHitsInALCT);

      } else {
        LOG_ERROR << cscTag << " Can not unpack ALCT Header or/and Trailer";
      }
    } else {
      /**   ALCT not found */

      if (getCSCHisto(h::CSC_ALCT_NUMBER_RATE, crateID, dmbID, mo)) {
        mo->Fill(0);
        int nALCT = (int)mo->GetBinContent(1);
        if (getCSCHisto(h::CSC_ALCT_NUMBER_EFFICIENCY, crateID, dmbID, mo))
          mo->SetBinContent(1, (float)(nALCT) / (float)(DMBEvents)*100.0);
      }

      if ((alct_dav > 0) && (getCSCHisto(h::CSC_DMB_FEB_UNPACKED_VS_DAV, crateID, dmbID, mo))) {
        mo->Fill(0.0, 1.0);
      }
    }

    /** ALCT and CLCT coinsidence */
    if (data.nclct() && data.nalct()) {
      CSCALCTHeader* alctHeader = data.alctHeader();

      if (alctHeader) {
        std::vector<CSCALCTDigi> alctsDatasTmp = alctHeader->ALCTDigis();
        std::vector<CSCALCTDigi> alctsDatas;

        for (uint32_t lct = 0; lct < alctsDatasTmp.size(); lct++) {
          if (alctsDatasTmp[lct].isValid())
            alctsDatas.push_back(alctsDatasTmp[lct]);
        }

        CSCTMBData* tmbData = data.tmbData();
        if (tmbData) {
          CSCTMBHeader* tmbHeader = tmbData->tmbHeader();
          if (tmbHeader) {
            if (getCSCHisto(h::CSC_TMB_BXN_VS_ALCT_BXN, crateID, dmbID, mo))
              mo->Fill(((int)(alctHeader->BXNCount())) % 256, ((int)(tmbHeader->BXNCount())) % 256);

            if (getCSCHisto(h::CSC_TMB_ALCT_BXN_DIFF, crateID, dmbID, mo)) {
              int clct_alct_bxn_diff = (int)(alctHeader->BXNCount() - tmbHeader->BXNCount());
              if (clct_alct_bxn_diff < -2048)
                mo->Fill(clct_alct_bxn_diff + 4096);
              else {
                if (clct_alct_bxn_diff > 2048)
                  mo->Fill(clct_alct_bxn_diff - 4096);
                else
                  mo->Fill(clct_alct_bxn_diff);
              }
              mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
            }

            if (getCSCHisto(h::CSC_TMB_L1A_VS_ALCT_L1A, crateID, dmbID, mo))
              mo->Fill((int)(alctHeader->L1Acc() % 256), (int)(tmbHeader->L1ANumber() % 256));

            if (getCSCHisto(h::CSC_TMB_ALCT_L1A_DIFF, crateID, dmbID, mo)) {
              int clct_alct_l1a_diff = (int)(tmbHeader->L1ANumber() - alctHeader->L1Acc());
              if (clct_alct_l1a_diff < -2048)
                mo->Fill(clct_alct_l1a_diff + 4096);
              else {
                if (clct_alct_l1a_diff > 2048)
                  mo->Fill(clct_alct_l1a_diff - 4096);
                else
                  mo->Fill(clct_alct_l1a_diff);
              }
              mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
            }
          } else {
            LOG_ERROR << cscTag << " Can not unpack TMB Header";
          }

        } else {
          LOG_ERROR << cscTag << " Can not unpack TMB Data";
        }
      } else {
        LOG_ERROR << cscTag << " Can not unpack ALCT Header";
      }
    }

    /** CLCT Found */
    if (data.nclct()) {
      /** LOG_WARN << "TMB CRC calc: 0x" << std::hex << data.tmbData().TMBCRCcalc() << " trailer: 0x" << std::hex << data.tmbData().tmbTrailer().crc22(); */

      CSCTMBData* tmbData = data.tmbData();
      if (tmbData) {
        CSCTMBHeader* tmbHeader = tmbData->tmbHeader();
        CSCTMBTrailer* tmbTrailer = tmbData->tmbTrailer();

        if (tmbHeader && tmbTrailer) {
          bool isRun3_df = false;
          bool isGEM_df = false;
          bool isTMB_hybrid_df = false;
          if (tmbHeader->FirmwareVersion() == 2020) {
            // revision code major part: 0x0 - Run3 df, 0x1 - is legacy Run2 df
            if (((tmbHeader->FirmwareRevision() >> 5) & 0xF) == 0x0)
              isRun3_df = true;
            if (((tmbHeader->FirmwareRevision() >> 9) & 0xF) == 0x3)
              isGEM_df = true;
            if (((tmbHeader->FirmwareRevision() >> 9) & 0xF) == 0x4)
              isTMB_hybrid_df = true;
            /** Summary plot for chambers with detected (O)TMB Run3 data format */
            if ((isTMB_hybrid_df || isRun3_df) && getEMUHisto(h::EMU_CSC_TMB_RUN3_DATA_FORMAT, mo)) {
              mo->SetBinContent(cscPosition, cscType + 1, 1);
            }
          }

          if (tmbHeader->FirmwareVersion() == 2020) {
            if (isRun3_df) {
              /** Summary plot for chambers with detected enabled (O)TMB Run3 CCLUT mode */
              if (getEMUHisto(h::EMU_CSC_TMB_RUN3_CCLUT_MODE, mo)) {
                if (tmbHeader->clct0_ComparatorCode() > 0)
                  mo->SetBinContent(cscPosition, cscType + 1, 1);
              }

              if (isGEM_df) {
                if (getCSCHisto(h::CSC_GEM_FIBERS_STATUS, crateID, dmbID, mo)) {
                  int gem_fibers = tmbHeader->gem_enabled_fibers();
                  for (unsigned i = 0; i < 4; i++) {
                    if ((gem_fibers >> i) & 0x1)
                      mo->Fill(i);
                  }
                }
                uint16_t gem_sync_status = tmbHeader->gem_sync_dataword();
                if (getCSCHisto(h::CSC_GEM_SYNC_STATUS, crateID, dmbID, mo)) {
                  for (int i = 0; i < 15; i++) {
                    if ((gem_sync_status >> i) & 0x1)
                      mo->Fill(i);
                  }
                }
                uint16_t gem_timing_status = tmbHeader->gem_timing_dataword();
                if (getCSCHisto(h::CSC_GEM_NUM_COPADS, crateID, dmbID, mo)) {
                  // For GEM firmware after rev0. 4-bits num_copad
                  mo->Fill(gem_timing_status & 0xF);
                }

                if (tmbData->hasGEM()) {
                  CSCGEMData* gemData = tmbData->gemData();
                  if (gemData != nullptr) {
                    bool isGEM_hits_found = false;
                    std::vector<CSCCorrelatedLCTDigi> corr_lctsDatasTmp = tmbHeader->CorrelatedLCTDigis(cid.rawId());
                    for (int i = 0; i < gemData->numGEMs(); i++) {
                      std::vector<GEMPadDigiCluster> gemDigis = gemData->digis(i);
                      if (getCSCHisto(h::CSC_GEM_NUM_CLUSTERS, crateID, dmbID, mo)) {
                        mo->Fill(gemDigis.size());
                      }

                      if (!gemDigis.empty()) {
                        isGEM_hits_found = true;
                        for (unsigned digi = 0; digi < gemDigis.size(); digi++) {
                          if (gemDigis[digi].isValid()) {
                            std::vector<uint16_t> pads_hits = gemDigis[digi].pads();
                            int hits_timebin = gemDigis[digi].bx();
                            if (getCSCHisto(((i == 0) ? h::CSC_GEM_GEMA_CLUSTER_SIZE : h::CSC_GEM_GEMB_CLUSTER_SIZE),
                                            crateID,
                                            dmbID,
                                            mo)) {
                              mo->Fill(pads_hits.size());
                            }
                            if (!pads_hits.empty()) {
                              if (getCSCHisto(((i == 0) ? h::CSC_GEM_GEMA_PADS_CLUSTER_SIZE
                                                        : h::CSC_GEM_GEMB_PADS_CLUSTER_SIZE),
                                              crateID,
                                              dmbID,
                                              mo)) {
                                // int npad = (7 - (pads_hits[0] / 192)) * 192 + pads_hits[0] % 192;
                                int npad = pads_hits[0];  // no eta recode
                                mo->Fill(npad, pads_hits.size());
                              }

                              if (getCSCHisto(
                                      ((i == 0) ? h::CSC_GEM_GEMA_BX_DISTRIBUTION : h::CSC_GEM_GEMB_BX_DISTRIBUTION),
                                      crateID,
                                      dmbID,
                                      mo)) {
                                mo->Fill(hits_timebin);
                              }

                              /// Fill summary GEM VFATs occupancies plots for endcaps
                              if ((cid.endcap() == 1) && getEMUHisto(h::EMU_GEM_PLUS_ENDCAP_VFAT_OCCUPANCY, mo)) {
                                int vfat = (pads_hits[0] / 192) + ((pads_hits[0] % 192) / 64) * 8;
                                mo->Fill((cscPosition)*2 + i - 1, vfat);
                              }

                              if ((cid.endcap() == 2) && getEMUHisto(h::EMU_GEM_MINUS_ENDCAP_VFAT_OCCUPANCY, mo)) {
                                int vfat = (pads_hits[0] / 192) + ((pads_hits[0] % 192) / 64) * 8;
                                mo->Fill((cscPosition)*2 + i - 1, vfat);
                              }

                              for (unsigned pad = 0; pad < pads_hits.size(); pad++) {
                                // int npad = (7 - (pads_hits[pad] / 192)) * 192 + pads_hits[pad] % 192;
                                int npad = pads_hits[pad];  // no eta recode
                                if (getCSCHisto(
                                        ((i == 0) ? h::CSC_GEM_GEMA_HITS : h::CSC_GEM_GEMB_HITS), crateID, dmbID, mo)) {
                                  mo->Fill(npad);
                                }

                                if (getCSCHisto(
                                        ((i == 0) ? h::CSC_GEM_GEMA_HITS_IN_TIME : h::CSC_GEM_GEMB_HITS_IN_TIME),
                                        crateID,
                                        dmbID,
                                        mo)) {
                                  mo->Fill(npad, hits_timebin);
                                }
                                if (getCSCHisto(((i == 0) ? h::CSC_GEM_GEMA_HITS_IN_TIME_PROFILE
                                                          : h::CSC_GEM_GEMB_HITS_IN_TIME_PROFILE),
                                                crateID,
                                                dmbID,
                                                mo)) {
                                  mo->Fill(npad, hits_timebin);
                                }

                                if (getCSCHisto(((i == 0) ? h::CSC_GEM_GEMA_VFAT_HITS_IN_TIME
                                                          : h::CSC_GEM_GEMB_VFAT_HITS_IN_TIME),
                                                crateID,
                                                dmbID,
                                                mo)) {
                                  int vfat = (pads_hits[0] / 192) + ((pads_hits[0] % 192) / 64) * 8;
                                  mo->Fill(vfat, hits_timebin);
                                }
                                if (getCSCHisto(((i == 0) ? h::CSC_GEM_GEMA_VFAT_HITS_IN_TIME_PROFILE
                                                          : h::CSC_GEM_GEMB_VFAT_HITS_IN_TIME_PROFILE),
                                                crateID,
                                                dmbID,
                                                mo)) {
                                  int vfat = (pads_hits[0] / 192) + ((pads_hits[0] % 192) / 64) * 8;
                                  mo->Fill(vfat, hits_timebin);
                                }
                              }
                            }  // pads_hits.size() > 0
                          }
                        }
                      }
                      for (unsigned ieta = 0; ieta < 8; ieta++) {
                        std::vector<GEMPadDigiCluster> gemEtaDigis =
                            gemData->etaDigis(i, ieta, tmbHeader->ALCTMatchTime());
                        if (!gemEtaDigis.empty()) {
                          for (unsigned digi = 0; digi < gemEtaDigis.size(); digi++) {
                            if (gemEtaDigis[digi].isValid()) {
                              std::vector<uint16_t> pads_hits = gemEtaDigis[digi].pads();
                              for (unsigned pad = 0; pad < pads_hits.size(); pad++) {
                                if (getCSCHisto(
                                        ((i == 0) ? h::CSC_GEM_GEMA_HITS_W_ROLLS : h::CSC_GEM_GEMB_HITS_W_ROLLS),
                                        crateID,
                                        dmbID,
                                        mo)) {
                                  mo->Fill(pads_hits[pad], 7 - ieta);
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                    if (isGEM_hits_found) {
                      if (getCSCHisto(h::CSC_GEM_ALCT_MATCH, crateID, dmbID, mo)) {
                        mo->Fill((gem_timing_status >> 12) & 0x7);
                      }
                      if (getCSCHisto(h::CSC_GEM_CLCT_MATCH, crateID, dmbID, mo)) {
                        mo->Fill((gem_timing_status >> 8) & 0xF);
                      }
                    }
                  }
                }  // OTMB hasGEM
              }    // isGEM_df

              /// Summary occupancy plot for Anode HMT bits from OTMB
              if (getEMUHisto(h::EMU_CSC_ANODE_HMT_REPORTING, mo)) {
                if (tmbHeader->alctHMT() > 0)
                  mo->Fill(cscPosition, cscType);
              }

              /// Summary occupancy plot for Cathode HMT bits from OTMB
              if (getEMUHisto(h::EMU_CSC_CATHODE_HMT_REPORTING, mo)) {
                if (tmbHeader->clctHMT() > 0)
                  mo->Fill(cscPosition, cscType);
              }

              if (getCSCHisto(h::CSC_CLCT_RUN3_HMT_NHITS, crateID, dmbID, mo)) {
                mo->Fill(tmbHeader->hmt_nhits());
              }

              if (getCSCHisto(h::CSC_RUN3_CLCT_ALCT_HMT, crateID, dmbID, mo)) {
                mo->Fill(tmbHeader->clctHMT());
                mo->Fill(tmbHeader->alctHMT() + 4);
              }

              if (getCSCHisto(h::CSC_RUN3_CLCT_VS_ALCT_HMT_BITS, crateID, dmbID, mo)) {
                mo->Fill(tmbHeader->clctHMT(), tmbHeader->alctHMT());
              }

              if (getCSCHisto(h::CSC_CORR_LCT_RUN3_PATTERN_ID, crateID, dmbID, mo)) {
                mo->Fill(tmbHeader->run3_CLCT_patternID());
              }
            }
            /*
              LOG_DEBUG << "OTMB dump: " << "CLCT0_ComparatorCode: 0x" << std::hex << tmbHeader->CLCT0_ComparatorCode()
                        << ", CLCT0_xky: 0x" << std::hex << tmbHeader->CLCT0_xky()
                        << ", CLCT1_ComparatorCode: 0x" << std::hex << tmbHeader->CLCT1_ComparatorCode()
                        << ", CLCT1_xky: 0x" << std::hex << tmbHeader->CLCT1_xky()
                        << ", GEM_sync_dataword: 0x" << std::hex << tmbHeader->GEM_sync_dataword()
                        << ", GEM_timing_dataword: 0x" << std::hex << tmbHeader->GEM_timing_dataword()
                        << ",\n   HMT_nhits: 0x" << std::hex << tmbHeader->HMT_nhits()
                        << ", GEM_enabled_fibers: 0x" << std::hex << tmbHeader->GEM_enabled_fibers()
                        << ", GEM_fifo_tbins: " << std::dec << tmbHeader->GEM_fifo_tbins()
                        << ", GEM_fifo_pretrig: " << std::dec << tmbHeader->GEM_fifo_pretrig()
                        << ", GEM_zero_suppression: " << std::dec << tmbHeader->GEM_zero_suppress();
*/
          }
          CSCComparatorData* comparatorData = data.comparatorData();

          std::vector<CSCCLCTDigi> clctsDatasTmp = tmbHeader->CLCTDigis(cid.rawId());
          std::vector<CSCCLCTDigi> clctsDatas;

          for (uint32_t lct = 0; lct < clctsDatasTmp.size(); lct++) {
            if (clctsDatasTmp[lct].isValid()) {
              clctsDatas.push_back(clctsDatasTmp[lct]);
              if (clctsDatasTmp[lct].isRun3()) {
                if ((lct == 0) && getCSCHisto(h::CSC_CLCTXX_KEY_STRIP_TYPE, crateID, dmbID, lct, mo))
                  mo->Fill(tmbHeader->clct0_xky());
                if ((lct == 1) && getCSCHisto(h::CSC_CLCTXX_KEY_STRIP_TYPE, crateID, dmbID, lct, mo))
                  mo->Fill(tmbHeader->clct1_xky());
              }
            }
          }

          for (uint32_t lct = 0; lct < clctsDatas.size(); lct++) {
            /*
                  LOG_DEBUG << "CLCT Digis dump: "
                            << "CLCT" << lct << " isRun3:" << clctsDatasTmp[lct].isRun3()
                            << ", isValid: " << clctsDatasTmp[lct].isValid()
                            << ", getCFEB: " << clctsDatasTmp[lct].getCFEB()
                            << ", getKeyStrip: " << clctsDatasTmp[lct].getKeyStrip()
                            << ", getFractionalStrip: " <<  clctsDatasTmp[lct].getFractionalStrip()
                            << ", getQuartStrip: " << clctsDatasTmp[lct].getQuartStripBit()
                            << ", getEightStrip: " << clctsDatasTmp[lct].getEightStripBit()
                            << ",\n    getCompCode: 0x" << std::hex << clctsDatasTmp[lct].getCompCode()
                            << ", getQuality: " << clctsDatasTmp[lct].getQuality() << std::dec
                            << ", getPattern: " << clctsDatasTmp[lct].getPattern() << std::dec
                            << ", getBend: " << clctsDatasTmp[lct].getBend() << std::dec
                            << ", getSlope: " << clctsDatasTmp[lct].getSlope() << std::dec
                            << ", getFractionalSlope: " <<  clctsDatasTmp[lct].getFractionalSlope()
                            << ", getBX: " << clctsDatasTmp[lct].getBX()
                            << ", getRun3Pattern: 0x" << std::hex << clctsDatasTmp[lct].getRun3Pattern()
                            << std::dec;
*/
            if (clctsDatas[lct].isRun3()) {
              if (getCSCHisto(h::CSC_CLCTXX_COMPARATOR_CODE, crateID, dmbID, lct, mo)) {
                mo->Fill(clctsDatas[lct].getCompCode());
              }
              if (getCSCHisto(h::CSC_CLCTXX_RUN3_TO_RUN2_PATTERN, crateID, dmbID, lct, mo)) {
                int bend = clctsDatas[lct].getSlope() + ((clctsDatas[lct].getBend() & 0x1) << 4);
                mo->Fill(bend, clctsDatas[lct].getPattern());
              }
              if (getCSCHisto(h::CSC_CLCTXX_RUN3_BEND_VS_SLOPE, crateID, dmbID, lct, mo)) {
                mo->Fill(clctsDatas[lct].getSlope(), clctsDatas[lct].getBend());
              }
            }
          }

          /// Correlated LCTs processing
          std::vector<CSCCorrelatedLCTDigi> corr_lctsDatasTmp = tmbHeader->CorrelatedLCTDigis(cid.rawId());
          std::vector<CSCCorrelatedLCTDigi> corr_lctsDatas;

          for (uint32_t lct = 0; lct < corr_lctsDatasTmp.size(); lct++) {
            if (corr_lctsDatasTmp[lct].isValid()) {
              corr_lctsDatas.push_back(corr_lctsDatasTmp[lct]);
            }

            if (isGEM_df) {
              uint16_t gem_sync_status = tmbHeader->gem_sync_dataword();

              for (unsigned i = lct * 4; i < (lct * 4 + 4); i++) {
                if ((gem_sync_status >> i) & 0x1) {
                  if (corr_lctsDatasTmp[lct].isValid()) {
                    if (getCSCHisto(h::CSC_GEM_LCT_SYNC_STATUS, crateID, dmbID, mo)) {
                      mo->Fill(lct, i);  // Fill for valid LCT
                    }
                  }
                }
              }
            }

            if (getCSCHisto(h::CSC_CORR_LCT_CLCT_COMBINATION, crateID, dmbID, mo)) {
              mo->Fill(clctsDatasTmp[lct].isValid() * 2 + lct, corr_lctsDatasTmp[lct].isValid() * 2 + lct);
            }

            if (lct == 0) {
              if (corr_lctsDatasTmp[lct].isRun3() || isTMB_hybrid_df) {
                /// Summary occupancy plot for combined HMT bits sent to MPC
                if (getEMUHisto(h::EMU_CSC_LCT_HMT_REPORTING, mo)) {
                  if (corr_lctsDatasTmp[lct].getHMT() > 0)
                    mo->Fill(cscPosition, cscType);
                }
                if (getCSCHisto(h::CSC_RUN3_HMT_DISTRIBUTION, crateID, dmbID, mo)) {
                  mo->Fill(corr_lctsDatasTmp[lct].getHMT());
                }
                if (getCSCHisto(h::CSC_CLCT_RUN3_HMT_NHITS_VS_HMT_BITS, crateID, dmbID, mo)) {
                  mo->Fill(tmbHeader->hmt_nhits(),
                           (corr_lctsDatasTmp[lct].getHMT() & 0x3));  // in-time Hadronic shower bits
                  mo->Fill(tmbHeader->hmt_nhits(),
                           ((corr_lctsDatasTmp[lct].getHMT() >> 2) & 0x3) + 4);  // out-of-time Hadronic shower bits
                }
                if (getCSCHisto(h::CSC_RUN3_HMT_COINCIDENCE_MATCH, crateID, dmbID, mo)) {
                  if ((corr_lctsDatasTmp[lct].getHMT() > 0) && (corr_lctsDatasTmp[lct].getHMT() <= 0xF)) {
                    mo->Fill(0);  // Run3 HMT is fired
                    if ((!clctsDatasTmp.empty()) && clctsDatasTmp[lct].isValid()) {
                      mo->Fill(2);  // Run3 HMT+CLCT match
                    }
                    if (corr_lctsDatasTmp[lct].isValid()) {
                      mo->Fill(3);  // Run3 HMT+LCT match
                    }
                  }
                }
                if (getCSCHisto(h::CSC_RUN3_HMT_ALCT_MATCH, crateID, dmbID, mo)) {
                  if ((corr_lctsDatasTmp[lct].getHMT() > 0) && (corr_lctsDatasTmp[lct].getHMT() <= 0xF)) {
                    mo->Fill(tmbHeader->hmt_ALCTMatchTime());
                  }
                }
                if (getCSCHisto(h::CSC_RUN3_HMT_HADRONIC_SHOWER, crateID, dmbID, mo)) {
                  mo->Fill(corr_lctsDatasTmp[lct].getHMT() & 0x3);               // in-time Hadronic shower bits
                  mo->Fill(((corr_lctsDatasTmp[lct].getHMT() >> 2) & 0x3) + 4);  // out-of-time Hadronic shower bits
                }
              }
            }
          }

          if (clctsDatasTmp[0].isValid() && corr_lctsDatasTmp[0].isValid() &&
              getCSCHisto(h::CSC_CORR_LCT0_VS_CLCT0_KEY_STRIP, crateID, dmbID, mo)) {
            mo->Fill(clctsDatasTmp[0].getKeyStrip(), corr_lctsDatasTmp[0].getStrip());
          }

          if (clctsDatasTmp[1].isValid() && corr_lctsDatasTmp[1].isValid() &&
              getCSCHisto(h::CSC_CORR_LCT1_VS_CLCT1_KEY_STRIP, crateID, dmbID, mo)) {
            mo->Fill(clctsDatasTmp[1].getKeyStrip(), corr_lctsDatasTmp[1].getStrip());
          }

          if (!corr_lctsDatas.empty()) {
            if (corr_lctsDatasTmp[0].isRun3()) {
              if (getCSCHisto(h::CSC_CORR_LCT0_VS_LCT1_RUN3_PATTERN, crateID, dmbID, mo)) {
                int lct1_pattern = corr_lctsDatasTmp[1].getRun3Pattern();
                if (!corr_lctsDatasTmp[1].isValid())
                  lct1_pattern = -1;
                mo->Fill(corr_lctsDatasTmp[0].getRun3Pattern(), lct1_pattern);
              }
            }
          }

          for (uint32_t lct = 0; lct < corr_lctsDatas.size(); lct++) {
            /*
                  LOG_DEBUG << "CorrelatedLCT Digis dump: "
                            << "CorrLCT" << lct << " isRun3:" << corr_lctsDatasTmp[lct].isRun3()
                            << ", isValid: " << corr_lctsDatasTmp[lct].isValid()
                            << ", getStrip: " << corr_lctsDatasTmp[lct].getStrip()
                            << ", getKeyWG: " << corr_lctsDatasTmp[lct].getKeyWG()
                            << ", getQuality: " << corr_lctsDatasTmp[lct].getQuality()
                            << ", getFractionalStrip: " <<  corr_lctsDatasTmp[lct].getFractionalStrip()
                            << ", getQuartStrip: " << corr_lctsDatasTmp[lct].getQuartStripBit()
                            << ", getEightStrip: " << corr_lctsDatasTmp[lct].getEightStripBit()
                            << ",\n getSlope: " << corr_lctsDatasTmp[lct].getSlope() << std::dec
                            << ", getBend: " << corr_lctsDatasTmp[lct].getBend()
                            << ", getBX: " << corr_lctsDatasTmp[lct].getBX()
                            << ", getPattern: 0x" << std::hex << corr_lctsDatasTmp[lct].getPattern()
                            << ", getRun3Pattern: 0x" << std::hex << corr_lctsDatasTmp[lct].getRun3Pattern()
                            // << ", getRun3PatternID: " << std::dec << corr_lctsDatasTmp[lct].getRun3PatternID()
                            << ", getHMT: " << corr_lctsDatasTmp[lct].getHMT()
			    << std::dec;
*/
            if (getCSCHisto(h::CSC_CORR_LCTXX_HITS_DISTRIBUTION, crateID, dmbID, lct, mo)) {
              mo->Fill(corr_lctsDatas[lct].getStrip(), corr_lctsDatas[lct].getKeyWG());
            }

            if (getCSCHisto(h::CSC_CORR_LCTXX_KEY_WG, crateID, dmbID, lct, mo)) {
              mo->Fill(corr_lctsDatas[lct].getKeyWG());
            }

            if (getCSCHisto(h::CSC_CORR_LCTXX_KEY_HALFSTRIP, crateID, dmbID, lct, mo)) {
              mo->Fill(corr_lctsDatas[lct].getStrip(2));
            }

            if (corr_lctsDatas[lct].isRun3()) {
              if (getCSCHisto(h::CSC_CORR_LCTXX_KEY_QUARTSTRIP, crateID, dmbID, lct, mo)) {
                mo->Fill(corr_lctsDatas[lct].getStrip(4));
              }

              if (getCSCHisto(h::CSC_CORR_LCTXX_KEY_EIGHTSTRIP, crateID, dmbID, lct, mo)) {
                mo->Fill(corr_lctsDatas[lct].getStrip(8));
              }

              if (getCSCHisto(h::CSC_CORR_LCTXX_RUN3_TO_RUN2_PATTERN, crateID, dmbID, lct, mo)) {
                int bend = corr_lctsDatas[lct].getSlope() + ((corr_lctsDatas[lct].getBend() & 0x1) << 4);
                mo->Fill(bend, corr_lctsDatas[lct].getPattern());
              }

              if (getCSCHisto(h::CSC_CORR_LCTXX_BEND_VS_SLOPE, crateID, dmbID, lct, mo)) {
                mo->Fill(corr_lctsDatas[lct].getSlope(), corr_lctsDatas[lct].getBend());
              }

              if (getCSCHisto(h::CSC_CORR_LCTXX_KEY_STRIP_TYPE, crateID, dmbID, lct, mo)) {
                int strip_type =
                    (corr_lctsDatas[lct].getQuartStripBit() << 1) + (corr_lctsDatas[lct].getEighthStripBit());
                mo->Fill(strip_type);
              }

              if ((cid.station() == 1) && (cid.ring() == 1)) {
                if (getCSCHisto(h::CSC_CORR_LCTXX_RUN3_ME11_QUALITY, crateID, dmbID, lct, mo)) {
                  mo->Fill(corr_lctsDatas[lct].getQuality());
                }
              } else {
                if (getCSCHisto(h::CSC_CORR_LCTXX_RUN3_QUALITY, crateID, dmbID, lct, mo)) {
                  mo->Fill(corr_lctsDatas[lct].getQuality());
                }
              }
            }
          }

          if (data.nalct()) {
            CSCALCTHeader* alctHeader = data.alctHeader();
            if (alctHeader) {
              std::vector<CSCALCTDigi> alctsDatasTmp = alctHeader->ALCTDigis();
              std::vector<CSCALCTDigi> alctsDatas;
              std::vector<uint32_t> keyWG_list;
              keyWG_list.clear();
              for (uint32_t lct = 0; lct < alctsDatasTmp.size(); lct++) {
                if (alctsDatasTmp[lct].isValid()) {
                  alctsDatas.push_back(alctsDatasTmp[lct]);
                  keyWG_list.push_back(alctsDatasTmp[lct].getKeyWG());
                }
              }

              if (!alctsDatas.empty() && getCSCHisto(h::CSC_RUN3_HMT_COINCIDENCE_MATCH, crateID, dmbID, mo) &&
                  corr_lctsDatasTmp[0].isRun3() && (corr_lctsDatasTmp[0].getHMT() > 0) &&
                  (corr_lctsDatasTmp[0].getHMT() <= 0xF) && alctsDatas[0].isValid()) {
                mo->Fill(1);  // Run3 HMT+ALCT match
              }

              for (uint32_t lct = 0; lct < corr_lctsDatasTmp.size(); lct++) {
                if (corr_lctsDatasTmp[lct].isValid()) {
                  for (uint32_t i = 0; i < keyWG_list.size(); i++) {
                    /// Find matching keyWG from ALCT keyWG list
                    if (corr_lctsDatasTmp[lct].getKeyWG() == keyWG_list[i]) {
                      if ((lct == 0) && getCSCHisto(h::CSC_CORR_LCT0_VS_ALCT0_KEY_WG, crateID, dmbID, mo))
                        mo->Fill(alctsDatas[i].getKeyWG(), corr_lctsDatasTmp[lct].getKeyWG());
                      if ((lct == 1) && getCSCHisto(h::CSC_CORR_LCT1_VS_ALCT1_KEY_WG, crateID, dmbID, mo))
                        mo->Fill(alctsDatas[i].getKeyWG(), corr_lctsDatasTmp[lct].getKeyWG());
                      if (getCSCHisto(h::CSC_CORR_LCT_VS_ALCT_DIGI_MATCH, crateID, dmbID, mo)) {
                        mo->Fill(lct, i);
                      }
                      continue;
                    }
                  }
                }
              }
            }
          }

          FEBunpacked = FEBunpacked + 1;
          tmb_unpacked = 1;

          if (getCSCHisto(h::CSC_ALCT_MATCH_TIME, crateID, dmbID, mo)) {
            mo->Fill(tmbHeader->ALCTMatchTime());
            double alct_match_mean = mo->getTH1()->GetMean();
            double alct_match_rms = mo->getTH1()->GetRMS();
            MonitorObject* mo1 = nullptr;

            if (cid.endcap() == 1) {
              if (cscPosition && getEMUHisto(h::EMU_CSC_ENDCAP_PLUS_ALCT_CLCT_MATCH_TIME, mo1)) {
                mo1->Fill(tmbHeader->ALCTMatchTime());
              }
            }

            if (cid.endcap() == 2) {
              if (cscPosition && getEMUHisto(h::EMU_CSC_ENDCAP_MINUS_ALCT_CLCT_MATCH_TIME, mo1)) {
                mo1->Fill(tmbHeader->ALCTMatchTime());
              }
            }

            if (cscPosition && getEMUHisto(h::EMU_CSC_ALCT_CLCT_MATCH_MEAN, mo1)) {
              mo1->SetBinContent(cscPosition, cscType + 1, alct_match_mean);
            }

            if (cscPosition && getEMUHisto(h::EMU_CSC_ALCT_CLCT_MATCH_RMS, mo1)) {
              mo1->SetBinContent(cscPosition, cscType + 1, alct_match_rms);
            }
          }

          if (getCSCHisto(h::CSC_LCT_MATCH_STATUS, crateID, dmbID, mo)) {
            if (tmbHeader->CLCTOnly())
              mo->Fill(0.0, 0.0);
            if (tmbHeader->ALCTOnly())
              mo->Fill(0.0, 1.0);
            if (tmbHeader->TMBMatch())
              mo->Fill(0.0, 2.0);
          }

          if (getCSCHisto(h::CSC_LCT0_MATCH_BXN_DIFFERENCE, crateID, dmbID, mo))
            mo->Fill(tmbHeader->Bxn0Diff());
          if (getCSCHisto(h::CSC_LCT1_MATCH_BXN_DIFFERENCE, crateID, dmbID, mo))
            mo->Fill(tmbHeader->Bxn1Diff());

          if ((tmb_dav > 0) && (getCSCHisto(h::CSC_DMB_FEB_UNPACKED_VS_DAV, crateID, dmbID, mo))) {
            mo->Fill(1.0, 0.0);
          }

          /**  Set number of CLCT-events to forth bin */
          if (getCSCHisto(h::CSC_CSC_RATE, crateID, dmbID, mo)) {
            mo->Fill(3);
            uint32_t CLCTEvent = (uint32_t)mo->GetBinContent(4);
            config->setChamberCounterValue(CLCT_TRIGGERS, crateID, dmbID, CLCTEvent);
            if (getCSCHisto(h::CSC_CSC_EFFICIENCY, crateID, dmbID, mo)) {
              if (config->getNEvents() > 0) {
                mo->SetBinContent(2, ((float)CLCTEvent / (float)(DMBEvents)*100.0));
                mo->SetEntries(DMBEvents);
              }
            }
          }

          if (getCSCHisto(h::CSC_CLCT_L1A, crateID, dmbID, mo))
            mo->Fill(tmbHeader->L1ANumber());

          /**  Use 6-bit L1A */
          if (getCSCHisto(h::CSC_CLCT_DMB_L1A_DIFF, crateID, dmbID, mo)) {
            int clct_dmb_l1a_diff = (int)((tmbHeader->L1ANumber() % 64) - dmbHeader->l1a24() % 64);
            if (clct_dmb_l1a_diff != 0)
              L1A_out_of_sync = true;
            if (clct_dmb_l1a_diff < -32)
              mo->Fill(clct_dmb_l1a_diff + 64);
            else {
              if (clct_dmb_l1a_diff >= 32)
                mo->Fill(clct_dmb_l1a_diff - 64);
              else
                mo->Fill(clct_dmb_l1a_diff);
            }
            mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
          }

          if (getCSCHisto(h::CSC_DMB_L1A_VS_CLCT_L1A, crateID, dmbID, mo))
            mo->Fill(tmbHeader->L1ANumber() % 256, dmbHeader->l1a24() % 256);

          if (getCSCHisto(h::CSC_CLCT_DMB_BXN_DIFF, crateID, dmbID, mo)) {
            int clct_dmb_bxn_diff = (int)(tmbHeader->BXNCount() % 64 - dmbHeader->bxn12() % 64);
            if (clct_dmb_bxn_diff < -32)
              mo->Fill(clct_dmb_bxn_diff + 64);
            else {
              if (clct_dmb_bxn_diff >= 32)
                mo->Fill(clct_dmb_bxn_diff - 64);
              else
                mo->Fill(clct_dmb_bxn_diff);
            }
            mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
          }

          if (getCSCHisto(h::CSC_CLCT_BXN, crateID, dmbID, mo))
            mo->Fill((int)(tmbHeader->BXNCount()));

          if (getCSCHisto(h::CSC_CLCT_BXN_VS_DMB_BXN, crateID, dmbID, mo))
            mo->Fill(tmbHeader->BXNCount() % 256, dmbHeader->bxn12() % 256);

          if (getCSCHisto(h::CSC_CLCT_NUMBER_RATE, crateID, dmbID, mo)) {
            mo->Fill(clctsDatas.size());
            int nCLCT = (int)mo->GetBinContent((int)(clctsDatas.size() + 1));
            if (getCSCHisto(h::CSC_CLCT_NUMBER, crateID, dmbID, mo))
              mo->SetBinContent((int)(clctsDatas.size() + 1), (float)(nCLCT) / (float)(DMBEvents)*100.0);
          }

          if (clctsDatas.size() == 1) {
            if (getCSCHisto(h::CSC_CLCT0_CLSSIFICATION, crateID, dmbID, mo)) {
              if (clctsDatas[0].getStripType())
                mo->Fill(0.0);
              else
                mo->Fill(1.0);
            }
          }

          if (clctsDatas.size() == 2) {
            if (getCSCHisto(h::CSC_CLCT1_VS_CLCT0_KEY_STRIP, crateID, dmbID, mo))
              mo->Fill(clctsDatas[0].getKeyStrip(), clctsDatas[1].getKeyStrip());
            if (getCSCHisto(h::CSC_CLCT0_CLCT1_CLSSIFICATION, crateID, dmbID, mo)) {
              if (clctsDatas[0].getStripType() && clctsDatas[1].getStripType())
                mo->Fill(0.0);
              if (clctsDatas[0].getStripType() && !clctsDatas[1].getStripType())
                mo->Fill(1.0);
              if (!clctsDatas[0].getStripType() && clctsDatas[1].getStripType())
                mo->Fill(2.0);
              if (!clctsDatas[0].getStripType() && !clctsDatas[1].getStripType())
                mo->Fill(3.0);
            }
          }

          if (getCSCHisto(h::CSC_TMB_WORD_COUNT, crateID, dmbID, mo))
            mo->Fill((int)(tmbTrailer->wordCount()));

          MonitorObject* mo_CSC_Plus_endcap_CLCT0_dTime = nullptr;
          getEMUHisto(h::EMU_CSC_ENDCAP_PLUS_CLCT0_DTIME, mo_CSC_Plus_endcap_CLCT0_dTime);

          MonitorObject* mo_CSC_Minus_endcap_CLCT0_dTime = nullptr;
          getEMUHisto(h::EMU_CSC_ENDCAP_MINUS_CLCT0_DTIME, mo_CSC_Minus_endcap_CLCT0_dTime);

          MonitorObject* mo_CSC_CLCT0_BXN_mean = nullptr;
          getEMUHisto(h::EMU_CSC_CLCT0_BXN_MEAN, mo_CSC_CLCT0_BXN_mean);

          MonitorObject* mo_CSC_CLCT0_BXN_rms = nullptr;
          getEMUHisto(h::EMU_CSC_CLCT0_BXN_RMS, mo_CSC_CLCT0_BXN_rms);

          for (uint32_t lct = 0; lct < clctsDatas.size(); lct++) {
            if (getCSCHisto(h::CSC_CLCTXX_BXN, crateID, dmbID, lct, mo))
              mo->Fill(clctsDatas[lct].getFullBX() % 64);

            int clct_dtime = clctsDatas[lct].getFullBX() - tmbHeader->BXNCount();
            if (clct_dtime > 0) {
              clct_dtime -= 3564;
            }

            int dTime = clct_dtime;

            if (lct == 0) {
              if (cid.endcap() == 1) {
                if (mo_CSC_Plus_endcap_CLCT0_dTime)
                  mo_CSC_Plus_endcap_CLCT0_dTime->Fill(dTime);
              }
              if (cid.endcap() == 2) {
                if (mo_CSC_Minus_endcap_CLCT0_dTime)
                  mo_CSC_Minus_endcap_CLCT0_dTime->Fill(dTime);
              }
            }

            if (getCSCHisto(h::CSC_CLCTXX_DTIME, crateID, dmbID, lct, mo)) {
              mo->Fill(dTime);
              mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");

              double dTime_mean = mo->getTH1()->GetMean();
              double dTime_rms = mo->getTH1()->GetRMS();

              // == For CLCT0 Fill Summary dTime Histograms
              if (lct == 0) {
                if (cscPosition && mo_CSC_CLCT0_BXN_mean) {
                  mo_CSC_CLCT0_BXN_mean->SetBinContent(cscPosition, cscType + 1, dTime_mean);
                }
                if (cscPosition && mo_CSC_CLCT0_BXN_rms) {
                  mo_CSC_CLCT0_BXN_rms->SetBinContent(cscPosition, cscType + 1, dTime_rms);
                }
              }
            }

            /* LOG_DEBUG << "LCT:" << lct << " Type:" << clctsDatas[lct].getStripType()
                      << " Strip:" << clctsDatas[lct].getKeyStrip();
             */

            if (clctsDatas[lct].getStripType()) {  // HalfStrip Type

              if (getCSCHisto(h::CSC_CLCTXX_KEYHALFSTRIP, crateID, dmbID, lct, mo))
                mo->Fill(clctsDatas[lct].getKeyStrip());

              /* === Run3 CSC-GEM Trigger data format */
              if (getCSCHisto(h::CSC_CLCTXX_KEY_QUARTSTRIP, crateID, dmbID, lct, mo))
                mo->Fill(clctsDatas[lct].getKeyStrip(4));
              if (getCSCHisto(h::CSC_CLCTXX_KEY_EIGHTSTRIP, crateID, dmbID, lct, mo))
                mo->Fill(clctsDatas[lct].getKeyStrip(8));
              /* === Run3 */

              if (getCSCHisto(h::CSC_CLCTXX_DTIME_VS_HALF_STRIP, crateID, dmbID, lct, mo)) {
                mo->Fill((int)(clctsDatas[lct].getKeyStrip()), clct_dtime);
              }

              if (getCSCHisto(h::CSC_CLCTXX_DTIME_PROFILE, crateID, dmbID, lct, mo)) {
                mo->Fill((int)(clctsDatas[lct].getKeyStrip()), clct_dtime);
              }

              if (getCSCHisto(h::CSC_CLCTXX_HALF_STRIP_PATTERN, crateID, dmbID, lct, mo)) {
                int pattern_clct = clctsDatas[lct].getPattern();
                /**  int pattern_clct = (int)((clctsDatas[lct].getPattern()>>1)&0x3); */
                /**  pattern_clct = Number of patterns in CLCT */
                /**  Last (left) bit is bend. Positive bend = 1, negative bend = 0 */
                double tbin = -1;

                switch (pattern_clct) {
                  case 0:
                    tbin = 0.;
                    break;
                  case 1:
                    tbin = 1.;
                    break;
                  case 2:
                    tbin = 2.;
                    break;
                  case 3:
                    tbin = 10.;
                    break;
                  case 4:
                    tbin = 3.;
                    break;
                  case 5:
                    tbin = 9.;
                    break;
                  case 6:
                    tbin = 4.;
                    break;
                  case 7:
                    tbin = 8.;
                    break;
                  case 8:
                    tbin = 5.;
                    break;
                  case 9:
                    tbin = 7.;
                    break;
                  case 10:
                    tbin = 6.;
                    break;
                }

                if (tbin >= 0)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), tbin);

                MonitorObject* mo1 = nullptr;
                if (getCSCHisto(h::CSC_CLCT_HALF_STRIP_PATTERN_DISTR, crateID, dmbID, lct, mo1))
                  mo1->Fill(tbin);
              }

              if (getCSCHisto(h::CSC_CLCTXX_HALF_STRIP_QUALITY, crateID, dmbID, lct, mo))
                mo->Fill((int)(clctsDatas[lct].getKeyStrip()), (int)(clctsDatas[lct].getQuality()));

              if (mo_EventDisplay) {
                mo_EventDisplay->SetBinContent(10, clctsDatas[lct].getKeyStrip(), clct_dtime);
                mo_EventDisplay->SetBinContent(11, clctsDatas[lct].getKeyStrip(), clctsDatas[lct].getQuality());
              }

              if (getCSCHisto(h::CSC_CLCTXX_HALF_STRIP_QUALITY_DISTR, crateID, dmbID, lct, mo)) {
                mo->Fill((int)(clctsDatas[lct].getQuality()));
                if (lct == 0) {
                  MonitorObject* mo1 = nullptr;
                  if (cscPosition && getEMUHisto(h::EMU_CSC_CLCT0_QUALITY, mo1)) {
                    mo1->SetBinContent(cscPosition, cscType + 1, mo->getTH1()->GetMean());
                  }
                }
              }

              if (getCSCHisto(h::CSC_CLCTXX_HALF_STRIP_QUALITY_PROFILE, crateID, dmbID, lct, mo))
                mo->Fill((int)(clctsDatas[lct].getKeyStrip()), (int)(clctsDatas[lct].getQuality()));

            } else {  // DiStrip Type

              LOG_INFO << "Entering block!";

              if (getCSCHisto(h::CSC_CLCTXX_KEYDISTRIP, crateID, dmbID, lct, mo))
                mo->Fill(clctsDatas[lct].getKeyStrip());
              else
                LOG_ERROR << "Not found h::CSC_CLCTXX_KEYDISTRIP = " << h::CSC_CLCTXX_KEYDISTRIP;

              if (lct == 0)
                clct_kewdistrip = clctsDatas[lct].getKeyStrip();

              if (getCSCHisto(h::CSC_CLCTXX_DTIME_VS_DISTRIP, crateID, dmbID, lct, mo)) {
                mo->Fill((int)(clctsDatas[lct].getKeyStrip()), clct_dtime);
              }

              if (getCSCHisto(h::CSC_CLCTXX_DISTRIP_PATTERN, crateID, dmbID, lct, mo)) {
                int pattern_clct = (int)((clctsDatas[lct].getPattern() >> 1) & 0x3);
                /**  pattern_clct = Number of patterns in CLCT */
                /**  Last (left) bit is bend. Positive bend = 1, negative bend = 0 */
                if (pattern_clct == 1)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 7.0);
                if (pattern_clct == 3)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 6.0);
                if (pattern_clct == 5)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 5.0);
                if (pattern_clct == 7)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 4.0);
                if (pattern_clct == 6)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 3.0);
                if (pattern_clct == 4)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 2.0);
                if (pattern_clct == 2)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 1.0);
                if (pattern_clct == 0)
                  mo->Fill(clctsDatas[lct].getKeyStrip(), 0.0);
              }

              if (getCSCHisto(h::CSC_CLCTXX_DISTRIP_QUALITY, crateID, dmbID, lct, mo))
                mo->Fill((int)(clctsDatas[lct].getKeyStrip()), (int)(clctsDatas[lct].getQuality()));

              if (getCSCHisto(h::CSC_CLCTXX_DISTRIP_QUALITY_PROFILE, crateID, dmbID, lct, mo))
                mo->Fill((int)(clctsDatas[lct].getKeyStrip()), (int)(clctsDatas[lct].getQuality()));
            }
          }

          int N_CFEBs = tmbHeader->NCFEBs();
          int NumberOfLayersWithHitsInCLCT = 0;
          int NumberOfHalfStripsWithHitsInCLCT = 0;

          if (comparatorData && comparatorData->check()) {
            MonitorObject* mo_CFEB_Comparators_TimeSamples = nullptr;
            getCSCHisto(h::CSC_CFEB_COMPARATORS_TIMESAMPLES, crateID, dmbID, mo_CFEB_Comparators_TimeSamples);

            MonitorObject* mo_CSC_Plus_endcap_CFEB_Comparators_Time = nullptr;
            getEMUHisto(h::EMU_CSC_ENDCAP_PLUS_CFEB_COMPARATORS_TIME, mo_CSC_Plus_endcap_CFEB_Comparators_Time);

            MonitorObject* mo_CSC_Minus_endcap_CFEB_Comparators_Time = nullptr;
            getEMUHisto(h::EMU_CSC_ENDCAP_MINUS_CFEB_COMPARATORS_TIME, mo_CSC_Minus_endcap_CFEB_Comparators_Time);

            MonitorObject* mo_CSC_CFEB_Comparators_Time_mean = nullptr;
            getEMUHisto(h::EMU_CSC_CFEB_COMPARATORS_TIME_MEAN, mo_CSC_CFEB_Comparators_Time_mean);

            MonitorObject* mo_CSC_CFEB_Comparators_Time_rms = nullptr;
            getEMUHisto(h::EMU_CSC_CFEB_COMPARATORS_TIME_RMS, mo_CSC_CFEB_Comparators_Time_rms);

            for (int nCFEB = 0; nCFEB < N_CFEBs; ++nCFEB) {
              for (int nLayer = 1; nLayer <= 6; nLayer++) {
                int hstrip_previous = -1;
                int tbin_clct_previous = -1;
                bool CheckLayerCLCT = true;

                std::vector<CSCComparatorDigi> compOutData = comparatorData->comparatorDigis(nLayer, nCFEB);

                for (std::vector<CSCComparatorDigi>::iterator compOutDataItr = compOutData.begin();
                     compOutDataItr != compOutData.end();
                     ++compOutDataItr) {
                  /**  =VB= Fix to get right hafstrip */
                  int hstrip = 2 * (compOutDataItr->getStrip() - 1) + compOutDataItr->getComparator();
                  std::vector<int> tbins_clct = compOutDataItr->getTimeBinsOn();
                  int tbin_clct = (int)compOutDataItr->getTimeBin();

                  if (mo_EventDisplay) {
                    mo_EventDisplay->SetBinContent(nLayer + 11, hstrip, tbin_clct + 1);
                    setEmuEventDisplayBit(mo_Emu_EventDisplay_Anode, glChamberIndex, 160 + hstrip, nLayer - 1);
                    setEmuEventDisplayBit(mo_Emu_EventDisplay_XY, glChamberIndex, 160 + hstrip, nLayer - 1);
                  }

                  if (CheckLayerCLCT) {
                    NumberOfLayersWithHitsInCLCT = NumberOfLayersWithHitsInCLCT + 1;
                    CheckLayerCLCT = false;
                  }

                  for (uint32_t n = 0; n < tbins_clct.size(); n++) {
                    tbin_clct = tbins_clct[n];
                    if (hstrip != hstrip_previous ||
                        (tbin_clct != tbin_clct_previous + 1 && tbin_clct != tbin_clct_previous - 1)) {
                      if (getCSCHisto(h::CSC_CLCTTIME_LYXX, crateID, dmbID, nLayer, mo))
                        mo->Fill(hstrip, tbin_clct);

                      if (mo_CFEB_Comparators_TimeSamples)
                        mo_CFEB_Comparators_TimeSamples->Fill(tbin_clct);

                      if (cid.endcap() == 1) {
                        if (mo_CSC_Plus_endcap_CFEB_Comparators_Time)
                          mo_CSC_Plus_endcap_CFEB_Comparators_Time->Fill(tbin_clct);
                      }

                      if (cid.endcap() == 2) {
                        if (mo_CSC_Minus_endcap_CFEB_Comparators_Time)
                          mo_CSC_Minus_endcap_CFEB_Comparators_Time->Fill(tbin_clct);
                      }

                      if (getCSCHisto(h::CSC_CLCTTIME_LYXX_PROFILE, crateID, dmbID, nLayer, mo))
                        mo->Fill(hstrip, tbin_clct);

                      if (getCSCHisto(h::CSC_CLCT_LYXX_RATE, crateID, dmbID, nLayer, mo)) {
                        mo->Fill(hstrip);

                        double number_hstrip = mo->GetBinContent(hstrip + 1);
                        double Number_of_entries_CLCT = mo->GetEntries();

                        if (getCSCHisto(h::CSC_CLCT_LYXX_EFFICIENCY, crateID, dmbID, nLayer, mo)) {
                          mo->SetBinContent(hstrip + 1, number_hstrip);
                          if (DMBEvents > 0) {
                            double norm = (100.0 * Number_of_entries_CLCT) / ((double)(DMBEvents));
                            /**  if (norm < 1.0) norm=1; */
                            mo->SetNormFactor(norm);
                          } else {
                            mo->SetNormFactor(100.0);
                          }
                          mo->SetEntries(DMBEvents);
                        }
                      }
                    }

                    if (hstrip != hstrip_previous) {
                      NumberOfHalfStripsWithHitsInCLCT = NumberOfHalfStripsWithHitsInCLCT + 1;
                    }
                    hstrip_previous = hstrip;
                    tbin_clct_previous = tbin_clct;
                  }
                }
              }
            }

            if (mo_CFEB_Comparators_TimeSamples) {
              double comps_time_mean = mo_CFEB_Comparators_TimeSamples->getTH1()->GetMean();
              double comps_time_rms = mo_CFEB_Comparators_TimeSamples->getTH1()->GetRMS();

              if (cscPosition && mo_CSC_CFEB_Comparators_Time_mean) {
                mo_CSC_CFEB_Comparators_Time_mean->SetBinContent(cscPosition, cscType + 1, comps_time_mean);
              }
              if (cscPosition && mo_CSC_CFEB_Comparators_Time_rms) {
                mo_CSC_CFEB_Comparators_Time_rms->SetBinContent(cscPosition, cscType + 1, comps_time_rms);
              }
            }

          } else {
            LOG_ERROR << cscTag << " Can not unpack CLCT Data";
          }

          if (getCSCHisto(h::CSC_CLCT_NUMBER_OF_LAYERS_WITH_HITS, crateID, dmbID, mo)) {
            mo->Fill(NumberOfLayersWithHitsInCLCT);
            MonitorObject* mo1 = nullptr;
            if (cscPosition && getEMUHisto(h::EMU_CSC_CLCT_PLANES_WITH_HITS, mo1)) {
              mo1->SetBinContent(cscPosition, cscType + 1, mo->getTH1()->GetMean());
            }
          }

          if (getCSCHisto(h::CSC_CLCT_NUMBER_OF_HALFSTRIPS_WITH_HITS, crateID, dmbID, mo))
            mo->Fill(NumberOfHalfStripsWithHitsInCLCT);
        } else {
          LOG_ERROR << cscTag << " Can not unpack TMB Header or/and Trailer";
        }
      } else {
        LOG_ERROR << cscTag << " Can not unpack TMB Data";
      }

    } else {
      /**   CLCT not found */

      if (getCSCHisto(h::CSC_CLCT_NUMBER_RATE, crateID, dmbID, mo)) {
        mo->Fill(0);
        int nCLCT = (int)mo->GetBinContent(1);
        if (getCSCHisto(h::CSC_CLCT_NUMBER, crateID, dmbID, mo))
          mo->SetBinContent(1, (float)(nCLCT) / (float)(DMBEvents)*100.0);
      }
      if ((tmb_dav > 0) && (getCSCHisto(h::CSC_DMB_FEB_UNPACKED_VS_DAV, crateID, dmbID, mo))) {
        mo->Fill(1.0, 1.0);
      }
    }

    /**  CFEB found */
    int NumberOfUnpackedCFEBs = 0;
    const int N_CFEBs = nCFEBs, N_Samples = 16, N_Layers = 6, N_Strips = 16, nStrips = nCFEBs * N_Strips;
    int ADC = 0, OutOffRange, Threshold = 30;
    /**  bool DebugCFEB = false; */
    int Pedestal[N_CFEBs][6][16];
    memset(Pedestal, 0, sizeof(Pedestal));
#ifdef __clang__
    std::vector<std::array<std::array<std::pair<int, int>, 6>, 16>> CellPeak(N_CFEBs);
    std::fill(CellPeak.begin(), CellPeak.end(), std::array<std::array<std::pair<int, int>, 6>, 16>{});
#else
    std::pair<int, int> CellPeak[N_CFEBs][6][16];
    memset(CellPeak, 0, sizeof(CellPeak));
#endif
    bool CheckCFEB = true;
    /** --------------B */
    float Clus_Sum_Charge;
    int TrigTime, L1APhase, UnpackedTrigTime, LCTPhase, SCA_BLK, NmbTimeSamples;
    int FreeCells, LCT_Pipe_Empty, LCT_Pipe_Full, LCT_Pipe_Count, L1_Pipe_Empty, L1_Pipe_Full, Buffer_Count;
    /** --------------E */

    bool CheckThresholdStripInTheLayer[6][nStrips];
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < nStrips; j++)
        CheckThresholdStripInTheLayer[i][j] = true;
    }

    bool CheckOutOffRangeStripInTheLayer[6][nStrips];
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < nStrips; j++)
        CheckOutOffRangeStripInTheLayer[i][j] = true;
    }

    /** --------------B */
    float cscdata[N_CFEBs * 16][N_Samples][N_Layers];
    int SCABlockData[N_CFEBs * 16][N_Samples][N_Layers];
    memset(cscdata, 0, sizeof(cscdata));
    memset(SCABlockData, 0, sizeof(SCABlockData));
    /** --------------E */

    char hbuf[255];
    memset(hbuf, 0, sizeof(hbuf));

    MonitorObject* mo_CFEB_SCA_CellPeak_Time = nullptr;
    getCSCHisto(h::CSC_CFEB_SCA_CELLPEAK_TIME, crateID, dmbID, mo_CFEB_SCA_CellPeak_Time);

    MonitorObject* mo_CSC_Plus_endcap_CFEB_SCA_CellPeak_Time = nullptr;
    getEMUHisto(h::EMU_CSC_PLUS_ENDCAP_CFEB_SCA_CELLPEAK_TIME, mo_CSC_Plus_endcap_CFEB_SCA_CellPeak_Time);

    MonitorObject* mo_CSC_Minus_endcap_CFEB_SCA_CellPeak_Time = nullptr;
    getEMUHisto(h::EMU_CSC_MINUS_ENDCAP_CFEB_SCA_CELLPEAK_TIME, mo_CSC_Minus_endcap_CFEB_SCA_CellPeak_Time);

    MonitorObject* mo_CSC_CFEB_SCA_CellPeak_Time_mean = nullptr;
    getEMUHisto(h::EMU_CSC_CFEB_SCA_CELLPEAK_TIME_MEAN, mo_CSC_CFEB_SCA_CellPeak_Time_mean);

    MonitorObject* mo_CSC_CFEB_SCA_CellPeak_Time_rms = nullptr;
    getEMUHisto(h::EMU_CSC_CFEB_SCA_CELLPEAK_TIME_RMS, mo_CSC_CFEB_SCA_CellPeak_Time_rms);

    for (int nCFEB = 0; nCFEB < N_CFEBs; ++nCFEB) {
      //       cfebData[nCFEB] = data.cfebData(nCFEB);
      if (data.cfebData(nCFEB) != nullptr) {
        if (!data.cfebData(nCFEB)->check())
          continue;

        /**                         CFEB Found */
        FEBunpacked = FEBunpacked + 1;                      // Increment number of unpacked FED
        NumberOfUnpackedCFEBs = NumberOfUnpackedCFEBs + 1;  // Increment number of unpaked CFEB
        cfeb_unpacked = 1;
        if (CheckCFEB == true) {
          if (getCSCHisto(h::CSC_CSC_RATE, crateID, dmbID, mo)) {
            mo->Fill(4);
            uint32_t CFEBEvent = (uint32_t)mo->GetBinContent(5);
            config->setChamberCounterValue(CFEB_TRIGGERS, crateID, dmbID, CFEBEvent);
            if (getCSCHisto(h::CSC_CSC_EFFICIENCY, crateID, dmbID, mo)) {
              if (config->getNEvents() > 0) {
                mo->SetBinContent(3, ((float)CFEBEvent / (float)(DMBEvents)*100.0));
                mo->SetEntries((int)DMBEvents);
              }
            }
          }

          if ((cfeb_dav2 > 0) && (getCSCHisto(h::CSC_DMB_FEB_UNPACKED_VS_DAV, crateID, dmbID, mo))) {
            mo->Fill(2.0, 0.0);
          }
          CheckCFEB = false;
        }
        /** -------------B */
        NmbTimeSamples = (data.cfebData(nCFEB))->nTimeSamples();
        /** -------------E */

        /**  =VB= Optimizations for faster histogram object access  */
        MonitorObject* mo_CFEB_SCA_Block_Occupancy = nullptr;
        getCSCHisto(h::CSC_CFEBXX_SCA_BLOCK_OCCUPANCY, crateID, dmbID, nCFEB + 1, mo_CFEB_SCA_Block_Occupancy);
        MonitorObject* mo_CFEB_Free_SCA_Cells = nullptr;
        getCSCHisto(h::CSC_CFEBXX_FREE_SCA_CELLS, crateID, dmbID, nCFEB + 1, mo_CFEB_Free_SCA_Cells);
        MonitorObject* mo_CFEB_SCA_Blocks_Locked_by_LCTs = nullptr;
        getCSCHisto(
            h::CSC_CFEBXX_SCA_BLOCKS_LOCKED_BY_LCTS, crateID, dmbID, nCFEB + 1, mo_CFEB_SCA_Blocks_Locked_by_LCTs);
        MonitorObject* mo_CFEB_SCA_Blocks_Locked_by_LCTxL1 = nullptr;
        getCSCHisto(
            h::CSC_CFEBXX_SCA_BLOCKS_LOCKED_BY_LCTXL1, crateID, dmbID, nCFEB + 1, mo_CFEB_SCA_Blocks_Locked_by_LCTxL1);
        MonitorObject* mo_CFEB_DMB_L1A_diff = nullptr;
        getCSCHisto(h::CSC_CFEBXX_DMB_L1A_DIFF, crateID, dmbID, nCFEB + 1, mo_CFEB_DMB_L1A_diff);

        for (int nLayer = 1; nLayer <= N_Layers; ++nLayer) {
          /**   =VB= Optimizations for faster histogram object access */
          MonitorObject* mo_CFEB_Out_Off_Range_Strips = nullptr;
          getCSCHisto(h::CSC_CFEB_OUT_OFF_RANGE_STRIPS_LYXX, crateID, dmbID, nLayer, mo_CFEB_Out_Off_Range_Strips);
          MonitorObject* mo_CFEB_Active_Samples_vs_Strip = nullptr;
          getCSCHisto(
              h::CSC_CFEB_ACTIVE_SAMPLES_VS_STRIP_LYXX, crateID, dmbID, nLayer, mo_CFEB_Active_Samples_vs_Strip);
          MonitorObject* mo_CFEB_Active_Samples_vs_Strip_Profile = nullptr;
          getCSCHisto(h::CSC_CFEB_ACTIVE_SAMPLES_VS_STRIP_LYXX_PROFILE,
                      crateID,
                      dmbID,
                      nLayer,
                      mo_CFEB_Active_Samples_vs_Strip_Profile);
          MonitorObject* mo_CFEB_ActiveStrips = nullptr;
          getCSCHisto(h::CSC_CFEB_ACTIVESTRIPS_LYXX, crateID, dmbID, nLayer, mo_CFEB_ActiveStrips);
          MonitorObject* mo_CFEB_SCA_Cell_Peak = nullptr;
          getCSCHisto(h::CSC_CFEB_SCA_CELL_PEAK_LY_XX, crateID, dmbID, nLayer, mo_CFEB_SCA_Cell_Peak);

          MonitorObject* mo_CFEB_Pedestal_withEMV_Sample = nullptr;
          getCSCHisto(
              h::CSC_CFEB_PEDESTAL_WITHEMV_SAMPLE_01_LYXX, crateID, dmbID, nLayer, mo_CFEB_Pedestal_withEMV_Sample);
          MonitorObject* mo_CFEB_Pedestal_withRMS_Sample = nullptr;
          getCSCHisto(
              h::CSC_CFEB_PEDESTAL_WITHRMS_SAMPLE_01_LYXX, crateID, dmbID, nLayer, mo_CFEB_Pedestal_withRMS_Sample);
          MonitorObject* mo_CFEB_PedestalRMS_Sample = nullptr;
          getCSCHisto(h::CSC_CFEB_PEDESTALRMS_SAMPLE_01_LYXX, crateID, dmbID, nLayer, mo_CFEB_PedestalRMS_Sample);

          for (int nSample = 0; nSample < NmbTimeSamples; ++nSample) {
            if (timeSlice(data, nCFEB, nSample) == nullptr) {
              LOG_WARN << "CFEB" << nCFEB << " nSample: " << nSample << " - B-Word";
              continue;
            }

            if (mo_CFEB_DMB_L1A_diff && !fCloseL1As) {
              int cfeb_dmb_l1a_diff =
                  (int)((timeSlice(data, nCFEB, nSample)->get_L1A_number()) - dmbHeader->l1a24() % 64);
              if (cfeb_dmb_l1a_diff != 0) {
                L1A_out_of_sync = true;
              }
              if (cfeb_dmb_l1a_diff < -32)
                mo->Fill(cfeb_dmb_l1a_diff + 64);
              else {
                if (cfeb_dmb_l1a_diff >= 32)
                  mo->Fill(cfeb_dmb_l1a_diff - 64);
                else
                  mo_CFEB_DMB_L1A_diff->Fill(cfeb_dmb_l1a_diff);
              }
              mo_CFEB_DMB_L1A_diff->SetAxisRange(
                  0.1, 1.1 * (1.0 + mo_CFEB_DMB_L1A_diff->GetBinContent(mo_CFEB_DMB_L1A_diff->GetMaximumBin())), "Y");
            }

            /** --------------B */
            FreeCells = timeSlice(data, nCFEB, nSample)->get_n_free_sca_blocks();
            LCT_Pipe_Empty = timeSlice(data, nCFEB, nSample)->get_lctpipe_empty();
            LCT_Pipe_Full = timeSlice(data, nCFEB, nSample)->get_lctpipe_full();
            LCT_Pipe_Count = timeSlice(data, nCFEB, nSample)->get_lctpipe_count();
            L1_Pipe_Empty = timeSlice(data, nCFEB, nSample)->get_l1pipe_empty();
            L1_Pipe_Full = timeSlice(data, nCFEB, nSample)->get_l1pipe_full();
            Buffer_Count = timeSlice(data, nCFEB, nSample)->get_buffer_count();

            SCA_BLK = (int)(timeSlice(data, nCFEB, nSample)->scaControllerWord(nLayer).sca_blk);

            for (int nStrip = 0; nStrip < N_Strips; ++nStrip) {
              SCABlockData[nCFEB * 16 + nStrip][nSample][nLayer - 1] = SCA_BLK;
            }

            /**  SCA Block Occupancy Histograms */
            if (mo_CFEB_SCA_Block_Occupancy)
              mo_CFEB_SCA_Block_Occupancy->Fill(SCA_BLK);

            /**  Free SCA Cells */
            if (mo_CFEB_Free_SCA_Cells) {
              if (timeSlice(data, nCFEB, nSample)->scaControllerWord(nLayer).sca_full == 1)
                mo_CFEB_Free_SCA_Cells->Fill(-1);
              mo_CFEB_Free_SCA_Cells->Fill(FreeCells);
            }

            /**  Number of SCA Blocks Locked by LCTs */
            if (mo_CFEB_SCA_Blocks_Locked_by_LCTs) {
              if (LCT_Pipe_Empty == 1)
                mo_CFEB_SCA_Blocks_Locked_by_LCTs->Fill(-0.5);
              if (LCT_Pipe_Full == 1)
                mo_CFEB_SCA_Blocks_Locked_by_LCTs->Fill(16.5);
              mo_CFEB_SCA_Blocks_Locked_by_LCTs->Fill(LCT_Pipe_Count);
            }

            /**  Number of SCA Blocks Locked by LCTxL1 */
            if (mo_CFEB_SCA_Blocks_Locked_by_LCTxL1) {
              if (L1_Pipe_Empty == 1)
                mo_CFEB_SCA_Blocks_Locked_by_LCTxL1->Fill(-0.5);
              if (L1_Pipe_Full == 1)
                mo_CFEB_SCA_Blocks_Locked_by_LCTxL1->Fill(31.5);
              mo_CFEB_SCA_Blocks_Locked_by_LCTxL1->Fill(Buffer_Count);
            }

            /** --------------E */
            if (nSample == 0 && nLayer == 1) {
              TrigTime = (int)(timeSlice(data, nCFEB, nSample)->scaControllerWord(nLayer).trig_time);
              int k = 1;
              while (((TrigTime >> (k - 1)) & 0x1) != 1 && k <= 8) {
                k = k + 1;
              }
              L1APhase = (int)((timeSlice(data, nCFEB, nSample)->scaControllerWord(nLayer).l1a_phase) & 0x1);
              UnpackedTrigTime = ((k << 1) & 0xE) + L1APhase;

              if (getCSCHisto(h::CSC_CFEBXX_L1A_SYNC_TIME, crateID, dmbID, nCFEB + 1, mo))
                mo->Fill((int)UnpackedTrigTime);
              LCTPhase = (int)((timeSlice(data, nCFEB, nSample)->scaControllerWord(nLayer).lct_phase) & 0x1);

              if (getCSCHisto(h::CSC_CFEBXX_LCT_PHASE_VS_L1A_PHASE, crateID, dmbID, nCFEB + 1, mo))
                mo->Fill(LCTPhase, L1APhase);

              if (getCSCHisto(h::CSC_CFEBXX_L1A_SYNC_TIME_VS_DMB, crateID, dmbID, nCFEB + 1, mo))
                mo->Fill((int)(dmbHeader->dmbCfebSync()), (int)UnpackedTrigTime);

              if (getCSCHisto(h::CSC_CFEBXX_L1A_SYNC_TIME_DMB_DIFF, crateID, dmbID, nCFEB + 1, mo)) {
                int cfeb_dmb_L1A_sync_time = (int)(dmbHeader->dmbCfebSync()) - (int)UnpackedTrigTime;
                if (cfeb_dmb_L1A_sync_time < -8)
                  mo->Fill(cfeb_dmb_L1A_sync_time + 16);
                else {
                  if (cfeb_dmb_L1A_sync_time >= 8)
                    mo->Fill(cfeb_dmb_L1A_sync_time - 16);
                  else
                    mo->Fill(cfeb_dmb_L1A_sync_time);
                }
                mo->SetAxisRange(0.1, 1.1 * (1.0 + mo->GetBinContent(mo->GetMaximumBin())), "Y");
              }
            }

            for (int nStrip = 1; nStrip <= N_Strips; ++nStrip) {
              ADC = (int)((timeSample(data, nCFEB, nSample, nLayer, nStrip)->adcCounts) & 0xFFF);
              OutOffRange = (int)((timeSample(data, nCFEB, nSample, nLayer, nStrip)->adcOverflow) & 0x1);

              if (nSample == 0) {  // nSample == 0
                CellPeak[nCFEB][nLayer - 1][nStrip - 1] = std::make_pair(nSample, ADC);
                Pedestal[nCFEB][nLayer - 1][nStrip - 1] = ADC;
              }

              if (OutOffRange == 1 && CheckOutOffRangeStripInTheLayer[nLayer - 1][nCFEB * 16 + nStrip - 1] == true) {
                if (mo_CFEB_Out_Off_Range_Strips)
                  mo_CFEB_Out_Off_Range_Strips->Fill((int)(nCFEB * 16 + nStrip - 1));
                CheckOutOffRangeStripInTheLayer[nLayer - 1][nCFEB * 16 + nStrip - 1] = false;
              }
              if (ADC - Pedestal[nCFEB][nLayer - 1][nStrip - 1] > Threshold && OutOffRange != 1) {
                if (mo_CFEB_Active_Samples_vs_Strip)
                  mo_CFEB_Active_Samples_vs_Strip->Fill((int)(nCFEB * 16 + nStrip - 1), nSample);

                if (mo_CFEB_Active_Samples_vs_Strip_Profile)
                  mo_CFEB_Active_Samples_vs_Strip_Profile->Fill((int)(nCFEB * 16 + nStrip - 1), nSample);

                if (CheckThresholdStripInTheLayer[nLayer - 1][nCFEB * 16 + nStrip - 1] == true) {
                  if (mo_CFEB_ActiveStrips)
                    mo_CFEB_ActiveStrips->Fill((int)(nCFEB * 16 + nStrip));
                  CheckThresholdStripInTheLayer[nLayer - 1][nCFEB * 16 + nStrip - 1] = false;
                }
                /** --------------B */
                if (ADC - Pedestal[nCFEB][nLayer - 1][nStrip - 1] > Threshold) {
                  cscdata[nCFEB * 16 + nStrip - 1][nSample][nLayer - 1] = ADC - Pedestal[nCFEB][nLayer - 1][nStrip - 1];
                }
                /** --------------E */
                if (ADC > CellPeak[nCFEB][nLayer - 1][nStrip - 1].second) {
                  CellPeak[nCFEB][nLayer - 1][nStrip - 1].first = nSample;
                  CellPeak[nCFEB][nLayer - 1][nStrip - 1].second = ADC;
                }
              }
              /**  continue; */
              /** --------------B */
              if (nSample == 1) {
                int channel_threshold = 40;
                if (std::abs(ADC - Pedestal[nCFEB][nLayer - 1][nStrip - 1]) < channel_threshold) {
                  if (mo_CFEB_Pedestal_withEMV_Sample)
                    mo_CFEB_Pedestal_withEMV_Sample->Fill((int)(nCFEB * 16 + nStrip - 1),
                                                          Pedestal[nCFEB][nLayer - 1][nStrip - 1]);

                  if (mo_CFEB_Pedestal_withRMS_Sample) {
                    mo_CFEB_Pedestal_withRMS_Sample->Fill((int)(nCFEB * 16 + nStrip - 1),
                                                          Pedestal[nCFEB][nLayer - 1][nStrip - 1]);

                    if (mo_CFEB_PedestalRMS_Sample) {
                      mo_CFEB_PedestalRMS_Sample->SetBinContent(
                          nCFEB * 16 + nStrip - 1, mo_CFEB_Pedestal_withRMS_Sample->GetBinError(nCFEB * 16 + nStrip));
                      mo_CFEB_PedestalRMS_Sample->SetBinError(nCFEB * 16 + nStrip - 1, 0.00000000001);
                    }
                  }
                }
                Pedestal[nCFEB][nLayer - 1][nStrip - 1] += ADC;
                Pedestal[nCFEB][nLayer - 1][nStrip - 1] /= 2;
              }
              /** --------------E */
            }
          }

          for (int nStrip = 1; nStrip <= N_Strips; ++nStrip) {
            if (mo_CFEB_SCA_Cell_Peak && CellPeak[nCFEB][nLayer - 1][nStrip - 1].first) {
              mo_CFEB_SCA_Cell_Peak->Fill((int)(nCFEB * 16 + nStrip - 1),
                                          CellPeak[nCFEB][nLayer - 1][nStrip - 1].first);
              if (mo_CFEB_SCA_CellPeak_Time) {
                mo_CFEB_SCA_CellPeak_Time->Fill(CellPeak[nCFEB][nLayer - 1][nStrip - 1].first);
              }

              if (mo_EventDisplay) {
                int peak_sample = CellPeak[nCFEB][nLayer - 1][nStrip - 1].first;
                int peak_adc = CellPeak[nCFEB][nLayer - 1][nStrip - 1].second;
                int pedestal = Pedestal[nCFEB][nLayer - 1][nStrip - 1];
                int peak_sca_charge = peak_adc - pedestal;

                if (peak_adc - pedestal > Threshold) {
                  if (peak_sample >= 1) {
                    peak_sca_charge +=
                        ((timeSample(data, nCFEB, peak_sample - 1, nLayer, nStrip)->adcCounts) & 0xFFF) - pedestal;
                  }
                  if (peak_sample < NmbTimeSamples - 1) {
                    peak_sca_charge +=
                        ((timeSample(data, nCFEB, peak_sample + 1, nLayer, nStrip)->adcCounts) & 0xFFF) - pedestal;
                  }
                  mo_EventDisplay->SetBinContent(nLayer + 17, nCFEB * 16 + nStrip - 1, peak_sca_charge);
                  setEmuEventDisplayBit(
                      mo_Emu_EventDisplay_Cathode, glChamberIndex, nCFEB * 16 + nStrip - 1, nLayer - 1);
                }
              }

              if (cid.endcap() == 1) {
                if (mo_CSC_Plus_endcap_CFEB_SCA_CellPeak_Time)
                  mo_CSC_Plus_endcap_CFEB_SCA_CellPeak_Time->Fill(CellPeak[nCFEB][nLayer - 1][nStrip - 1].first);
              }
              if (cid.endcap() == 2) {
                if (mo_CSC_Minus_endcap_CFEB_SCA_CellPeak_Time)
                  mo_CSC_Minus_endcap_CFEB_SCA_CellPeak_Time->Fill(CellPeak[nCFEB][nLayer - 1][nStrip - 1].first);
              }
            }
          }
        }
      }
    }

    // Fill Summary CFEB Raw Hits Timing Plots
    if (mo_CFEB_SCA_CellPeak_Time) {
      double cellpeak_time_mean = mo_CFEB_SCA_CellPeak_Time->getTH1()->GetMean();
      double cellpeak_time_rms = mo_CFEB_SCA_CellPeak_Time->getTH1()->GetRMS();
      if (cscPosition && mo_CSC_CFEB_SCA_CellPeak_Time_mean) {
        mo_CSC_CFEB_SCA_CellPeak_Time_mean->SetBinContent(cscPosition, cscType + 1, cellpeak_time_mean);
      }
      if (cscPosition && mo_CSC_CFEB_SCA_CellPeak_Time_rms) {
        mo_CSC_CFEB_SCA_CellPeak_Time_rms->SetBinContent(cscPosition, cscType + 1, cellpeak_time_rms);
      }
    }

    /** --------------B */
    /**  Refactored for performance (VR) */
    const int a = N_CFEBs * N_Strips;
    const int b = a * N_Samples;
    float Cathodes[b * N_Layers];
    for (int i = 0; i < N_Layers; ++i) {
      const int b1 = i * b;
      for (int j = 0; j < a; ++j) {
        const int b2 = b1 + j;
        for (int k = 0; k < N_Samples; ++k) {
          Cathodes[b2 + a * k] = cscdata[j][k][i];
        }
      }
    }

    std::vector<StripCluster> Clus;
    Clus.clear();
    StripClusterFinder ClusterFinder(N_Layers, N_Samples, N_CFEBs, N_Strips);

    for (int nLayer = 1; nLayer <= N_Layers; ++nLayer) {
      ClusterFinder.DoAction(nLayer - 1, Cathodes);
      Clus = ClusterFinder.getClusters();

      /**  Number of Clusters Histograms */
      if (getCSCHisto(h::CSC_CFEB_NUMBER_OF_CLUSTERS_LY_XX, crateID, dmbID, nLayer, mo)) {
        mo->Fill(Clus.size());
      }

      for (uint32_t u = 0; u < Clus.size(); u++) {
        Clus_Sum_Charge = 0.0;
        for (uint32_t k = 0; k < Clus[u].ClusterPulseMapHeight.size(); k++) {
          for (int n = Clus[u].LFTBNDTime; n < Clus[u].IRTBNDTime; n++) {
            Clus_Sum_Charge = Clus_Sum_Charge + Clus[u].ClusterPulseMapHeight[k].height_[n];
          }
        }

        /**  Clusters Charge Histograms */
        if (getCSCHisto(h::CSC_CFEB_CLUSTERS_CHARGE_LY_XX, crateID, dmbID, nLayer, mo)) {
          mo->Fill(Clus_Sum_Charge);
        }

        /**  Width of Clusters Histograms */
        if (getCSCHisto(h::CSC_CFEB_WIDTH_OF_CLUSTERS_LY_XX, crateID, dmbID, nLayer, mo)) {
          mo->Fill(Clus[u].IRTBNDStrip - Clus[u].LFTBNDStrip + 1);
        }

        /**  Cluster Duration Histograms */
        if (getCSCHisto(h::CSC_CFEB_CLUSTER_DURATION_LY_XX, crateID, dmbID, nLayer, mo)) {
          mo->Fill(Clus[u].IRTBNDTime - Clus[u].LFTBNDTime + 1);
        }
      }

      Clus.clear();

      /**  delete ClusterFinder; */
    }

    /** --------------E */

    /**  Fill Hisogram for Different Combinations of FEBs Unpacked vs DAV */
    if (getCSCHisto(h::CSC_DMB_FEB_COMBINATIONS_UNPACKED_VS_DAV, crateID, dmbID, mo)) {
      float feb_combination_unpacked = -1.0;
      if (alct_unpacked == 0 && tmb_unpacked == 0 && cfeb_unpacked == 0)
        feb_combination_unpacked = 0.0;
      if (alct_unpacked > 0 && tmb_unpacked == 0 && cfeb_unpacked == 0)
        feb_combination_unpacked = 1.0;
      if (alct_unpacked == 0 && tmb_unpacked > 0 && cfeb_unpacked == 0)
        feb_combination_unpacked = 2.0;
      if (alct_unpacked == 0 && tmb_unpacked == 0 && cfeb_unpacked > 0)
        feb_combination_unpacked = 3.0;
      if (alct_unpacked > 0 && tmb_unpacked > 0 && cfeb_unpacked == 0)
        feb_combination_unpacked = 4.0;
      if (alct_unpacked > 0 && tmb_unpacked == 0 && cfeb_unpacked > 0)
        feb_combination_unpacked = 5.0;
      if (alct_unpacked == 0 && tmb_unpacked > 0 && cfeb_unpacked > 0)
        feb_combination_unpacked = 6.0;
      if (alct_unpacked > 0 && tmb_unpacked > 0 && cfeb_unpacked > 0)
        feb_combination_unpacked = 7.0;
      mo->Fill(feb_combination_dav, feb_combination_unpacked);
    }

    if ((clct_kewdistrip > -1 && alct_keywg > -1) &&
        (getCSCHisto(h::CSC_CLCT0_KEYDISTRIP_VS_ALCT0_KEYWIREGROUP, crateID, dmbID, mo))) {
      mo->Fill(alct_keywg, clct_kewdistrip);
    }

    if (L1A_out_of_sync && cscPosition && getEMUHisto(h::EMU_CSC_L1A_OUT_OF_SYNC, mo)) {
      mo->Fill(cscPosition, cscType);
    }

    if (L1A_out_of_sync && getEMUHisto(h::EMU_DMB_L1A_OUT_OF_SYNC, mo)) {
      mo->Fill(crateID, dmbID);
    }
  }

}  // namespace cscdqm
