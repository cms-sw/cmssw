#ifndef EventFilter_CSCRawToDigi_CSCTMBHeader2020_CCLUT_h
#define EventFilter_CSCRawToDigi_CSCTMBHeader2020_CCLUT_h
#include "EventFilter/CSCRawToDigi/interface/CSCVTMBHeaderFormat.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

struct CSCTMBHeader2020_CCLUT : public CSCVTMBHeaderFormat {
  enum { NWORDS = 43 };
  CSCTMBHeader2020_CCLUT();
  CSCTMBHeader2020_CCLUT(const unsigned short* buf);
  void setEventInformation(const CSCDMBHeader& dmbHeader) override;

  uint16_t BXNCount() const override { return bits.bxnCount; }
  uint16_t ALCTMatchTime() const override { return bits.matchWin; }
  void setALCTMatchTime(uint16_t alctmatchtime) override { bits.matchWin = alctmatchtime & 0xF; }
  uint16_t CLCTOnly() const override { return bits.clctOnly; }
  uint16_t ALCTOnly() const override { return bits.alctOnly; }
  uint16_t TMBMatch() const override { return bits.tmbMatch; }
  uint16_t Bxn0Diff() const override { return 0; }
  uint16_t Bxn1Diff() const override { return 0; }
  uint16_t L1ANumber() const override { return bits.l1aNumber; }
  uint16_t NTBins() const override { return bits.nTBins; }
  uint16_t NCFEBs() const override { return bits.nCFEBs; }
  void setNCFEBs(uint16_t ncfebs) override { bits.nCFEBs = ncfebs & 0x7F; }
  uint16_t firmwareRevision() const override { return bits.firmRevCode; }
  uint16_t syncError() const override { return bits.syncError; }
  uint16_t syncErrorCLCT() const override { return bits.clct_sync_err; }
  uint16_t syncErrorMPC0() const override { return 0; }
  uint16_t syncErrorMPC1() const override { return 0; }
  uint16_t L1AMatchTime() const override { return bits.pop_l1a_match_win; }

  // == Run 3 CSC-GEM Trigger Format
  uint16_t clct0_ComparatorCode() const override { return bits.clct0_comparator_code; }
  uint16_t clct1_ComparatorCode() const override { return bits.clct1_comparator_code; }
  uint16_t clct0_xky() const override { return bits.clct0_xky; }
  uint16_t clct1_xky() const override { return bits.clct1_xky; }
  uint16_t hmt_nhits() const override {
    return ((bits.hmt_nhits_bit0 & 0x1) + ((bits.hmt_nhits_bit1 & 0x1) << 1) +
            ((bits.hmt_nhits_bits_high & 0x1F) << 2));
  }
  uint16_t hmt_ALCTMatchTime() const override { return bits.hmt_match_win; }
  uint16_t alctHMT() const override { return bits.anode_hmt; }
  uint16_t clctHMT() const override { return bits.cathode_hmt; }
  uint16_t gem_enabled_fibers() const override { return 0; }
  uint16_t gem_fifo_tbins() const override { return 0; }
  uint16_t gem_fifo_pretrig() const override { return 0; }
  uint16_t gem_zero_suppress() const override { return 0; }
  uint16_t gem_sync_dataword() const override { return 0; }
  uint16_t gem_timing_dataword() const override { return 0; }

  uint16_t run3_CLCT_patternID() const override {
    return (bits.MPC_Muon_clct_pattern_low | (bits.MPC_Muon_clct_pattern_bit5 << 4));
  }
  // ==

  ///returns CLCT digis
  std::vector<CSCCLCTDigi> CLCTDigis(uint32_t idlayer) override;
  ///returns CorrelatedLCT digis
  std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis(uint32_t idlayer) const override;
  ///returns lct HMT Shower digi
  CSCShowerDigi showerDigi(uint32_t idlayer) const override;
  ///returns anode HMT Shower digi
  CSCShowerDigi anodeShowerDigi(uint32_t idlayer) const override;
  ///returns cathode HMT Shower digi
  CSCShowerDigi cathodeShowerDigi(uint32_t idlayer) const override;

  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  unsigned short int sizeInWords() const override { return NWORDS; }

  unsigned short int NHeaderFrames() const override { return bits.nHeaderFrames; }
  /// returns the first data word
  unsigned short* data() override { return (unsigned short*)(&bits); }
  bool check() const override { return bits.e0bline == 0x6e0b; }

  /// for data packing
  void addCLCT0(const CSCCLCTDigi& digi) override;
  void addCLCT1(const CSCCLCTDigi& digi) override;
  void addALCT0(const CSCALCTDigi& digi) override;
  void addALCT1(const CSCALCTDigi& digi) override;
  void addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) override;
  void addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) override;
  void addShower(const CSCShowerDigi& digi) override;
  void addAnodeShower(const CSCShowerDigi& digi) override;
  void addCathodeShower(const CSCShowerDigi& digi) override;

  void swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2);

  void print(std::ostream& os) const override;

  struct {
    // 0
    unsigned b0cline : 16;
    unsigned bxnCount : 12, dduCode1 : 3, flag1 : 1;
    unsigned l1aNumber : 12, dduCode2 : 3, flag2 : 1;
    unsigned readoutCounter : 12, dduCode3 : 3, flag3 : 1;
    // 4
    unsigned boardID : 5, cscID : 4, runID : 4, stackOvf : 1, syncError : 1, flag4 : 1;
    unsigned nHeaderFrames : 6, fifoMode : 3, r_type : 2, l1atype : 2, hasBuf : 1, bufFull : 1, flag5 : 1;
    unsigned bd_status : 15, flag6 : 1;
    unsigned firmRevCode : 15, flag7 : 1;
    // 8
    unsigned bxnPreTrigger : 12, tmb_clct0_discard : 1, tmb_clct1_discard : 1, clock_lock_lost : 1, flag8 : 1;
    unsigned preTrigCounter : 15, flag9 : 1;
    unsigned clct0_comparator_code : 12, clct0_xky : 2, hmt_nhits_bit0 : 1, flag10 : 1;  // 12-bits comp code fw version
    unsigned clctCounterLow : 15, flag11 : 1;
    // 12
    unsigned clctCounterHigh : 15, flag12 : 1;
    unsigned trigCounter : 15, flag13 : 1;
    unsigned clct1_comparator_code : 12, clct1_xky : 2, hmt_nhits_bit1 : 1, flag14 : 1;  // 12-bits comp code fw version
    unsigned alctCounterLow : 15, flag15 : 1;
    // 16
    unsigned alctCounterHigh : 15, flag16 : 1;
    unsigned uptimeCounterLow : 15, flag17 : 1;
    unsigned uptimeCounterHigh : 15, flag18 : 1;
    unsigned nCFEBs : 3, nTBins : 5, fifoPretrig : 5, scopeExists : 1, vmeExists : 1, flag19 : 1;
    // 20
    unsigned hitThresh : 3, pidThresh : 4, nphThresh : 3, pid_thresh_postdrift : 4, staggerCSC : 1, flag20 : 1;
    unsigned triadPersist : 4, dmbThresh : 3, alct_delay : 4, clct_width : 4, flag21 : 1;
    unsigned trigSourceVect : 9, clct0_slope : 4, clct0_LR_bend : 1, clct1_LR_bend : 1, flag22 : 1;
    unsigned activeCFEBs : 5, readCFEBs : 5, pop_l1a_match_win : 4, aff_source : 1, flag23 : 1;
    // 24
    unsigned tmbMatch : 1, alctOnly : 1, clctOnly : 1, matchWin : 4, noALCT : 1, oneALCT : 1, oneCLCT : 1, twoALCT : 1,
        twoCLCT : 1, dupeALCT : 1, dupeCLCT : 1, lctRankErr : 1, flag24 : 1;
    unsigned clct0_valid : 1, clct0_quality : 3, clct0_shape : 4, clct0_key_low : 7, flag25 : 1;
    unsigned clct1_valid : 1, clct1_quality : 3, clct1_shape : 4, clct1_key_low : 7, flag26 : 1;
    unsigned clct0_key_high : 1, clct1_key_high : 1, clct_bxn : 2, clct_sync_err : 1, clct0Invalid : 1,
        clct1Invalid : 1, clct1Busy : 1, parity_err_cfeb_ram : 5, parity_err_rpc : 1, parity_err_summary : 1,
        flag27 : 1;
    // 28
    unsigned alct0Valid : 1, alct0Quality : 2, alct0Amu : 1, alct0Key : 7, clct1_slope : 4, flag28 : 1;
    unsigned alct1Valid : 1, alct1Quality : 2, alct1Amu : 1, alct1Key : 7, drift_delay : 2, bcb_read_enable : 1,
        hs_layer_trig : 1, flag29 : 1;
    unsigned hmt_nhits_bits_high : 5, alct_ecc_err : 2, cfeb_badbits_found : 5, cfeb_badbits_blocked : 1, alctCfg : 1,
        bx0_match : 1, flag30 : 1;
    unsigned MPC_Muon0_alct_key_wire : 7, MPC_Muon_clct_pattern_low : 4, MPC_Muon0_lct_quality : 3,
        MPC_Muon0_clct_QuarterStrip : 1, flag31 : 1;
    // 32
    unsigned MPC_Muon0_clct_key_halfstrip : 8, MPC_Muon0_clct_LR : 1, MPC_Muon0_clct_EighthStrip : 1,
        MPC_Muon_alct_bxn : 1, MPC_Muon0_clct_bx0 : 1, MPC_Muon0_clct_bend_low : 3, flag32 : 1;
    unsigned MPC_Muon1_alct_key_wire : 7, MPC_Muon_clct_pattern_bit5 : 1, MPC_Muon_HMT_high : 3,
        MPC_Muon1_lct_quality : 3, MPC_Muon1_clct_QuarterStrip : 1, flag33 : 1;
    unsigned MPC_Muon1_clct_key_halfstrip : 8, MPC_Muon1_clct_LR : 1, MPC_Muon1_clct_EighthStrip : 1,
        MPC_Muon_HMT_bit0 : 1, MPC_Muon1_clct_bx0 : 1, MPC_Muon1_clct_bend_low : 3, flag34 : 1;
    unsigned MPC_Muon0_lct_vpf : 1, MPC_Muon0_clct_bend_bit4 : 1, MPC_Muon1_lct_vpf : 1, MPC_Muon1_clct_bend_bit4 : 1,
        MPCDelay : 4, MPCAccept : 2, CFEBsEnabled : 5, flag35 : 1;
    // 36
    unsigned RPCList : 2, NRPCs : 2, RPCEnable : 1, fifo_tbins_rpc : 5, fifo_pretrig_rpc : 5, flag36 : 1;
    unsigned r_wr_buf_adr : 11, r_wr_buf_ready : 1, wr_buf_ready : 1, buf_q_full : 1, buf_q_empty : 1, flag37 : 1;
    unsigned r_buf_fence_dist : 11, buf_q_ovf_err : 1, buf_q_udf_err : 1, buf_q_adr_err : 1, buf_stalled : 1,
        flag38 : 1;
    unsigned buf_fence_cnt : 12, reverse_hs_csc : 1, reverse_hs_me1a : 1, reverse_hs_me1b : 1, flag39 : 1;
    // 40
    unsigned activeCFEBs_2 : 2, readCFEBs_2 : 2, cfeb_badbits_found_2 : 2, parity_err_cfeb_ram_2 : 2,
        CFEBsEnabled_2 : 2, buf_fence_cnt_is_peak : 1, mxcfeb : 1, trig_source_vec : 2, tmb_trig_pulse : 1, flag40 : 1;
    unsigned run3_trig_df : 1, gem_enable : 1, hmt_match_win : 4, tmb_alct_only_ro : 1, tmb_clct_only_ro : 1,
        tmb_match_ro : 1, tmb_trig_keep : 1, tmb_non_trig_keep : 1, cathode_hmt : 2, anode_hmt : 2, flag41 : 1;
    unsigned e0bline : 16;
  } bits;
};

#endif
