#ifndef EventFilter_CSCRawToDigi_CSCTMBHeader2006_h
#define EventFilter_CSCRawToDigi_CSCTMBHeader2006_h
#include "EventFilter/CSCRawToDigi/interface/CSCVTMBHeaderFormat.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

struct CSCTMBHeader2006 : public CSCVTMBHeaderFormat {
  enum { NWORDS = 27 };
  CSCTMBHeader2006();
  explicit CSCTMBHeader2006(const unsigned short* buf);
  void setEventInformation(const CSCDMBHeader& dmbHeader) override;

  uint16_t BXNCount() const override { return bits.bxnCount; }
  uint16_t ALCTMatchTime() const override { return bits.alctMatchTime; }
  void setALCTMatchTime(uint16_t alctmatchtime) override { bits.alctMatchTime = alctmatchtime & 0xF; }
  uint16_t CLCTOnly() const override { return bits.clctOnly; }
  uint16_t ALCTOnly() const override { return bits.alctOnly; }
  uint16_t TMBMatch() const override { return bits.tmbMatch; }
  uint16_t Bxn0Diff() const override { return bits.bxn0Diff; }
  uint16_t Bxn1Diff() const override { return bits.bxn1Diff; }
  uint16_t L1ANumber() const override { return bits.l1aNumber; }
  uint16_t NTBins() const override { return bits.nTBins; }
  uint16_t NCFEBs() const override { return bits.nCFEBs; }
  void setNCFEBs(uint16_t ncfebs) override { bits.nCFEBs = ncfebs & 0x1F; }
  uint16_t firmwareRevision() const override { return bits.firmRevCode; }
  uint16_t syncError() const override { return bits.syncError; }
  uint16_t syncErrorCLCT() const override { return (bits.clct0_sync_err | bits.clct1_sync_err); }
  uint16_t syncErrorMPC0() const override { return bits.MPC_Muon0_SyncErr_; }
  uint16_t syncErrorMPC1() const override { return bits.MPC_Muon1_SyncErr_; }
  uint16_t L1AMatchTime() const override { return bits.pop_l1a_match_win; }

  /// == Run 3 CSC-GEM Trigger Format
  uint16_t clct0_ComparatorCode() const override { return 0; }
  uint16_t clct1_ComparatorCode() const override { return 0; }
  uint16_t clct0_xky() const override { return 0; }
  uint16_t clct1_xky() const override { return 0; }
  uint16_t hmt_nhits() const override { return 0; }
  uint16_t hmt_ALCTMatchTime() const override { return 0; }
  uint16_t alctHMT() const override { return 0; }
  uint16_t clctHMT() const override { return 0; }
  uint16_t gem_enabled_fibers() const override { return 0; }
  uint16_t gem_fifo_tbins() const override { return 0; }
  uint16_t gem_fifo_pretrig() const override { return 0; }
  uint16_t gem_zero_suppress() const override { return 0; }
  uint16_t gem_sync_dataword() const override { return 0; }
  uint16_t gem_timing_dataword() const override { return 0; }
  uint16_t run3_CLCT_patternID() const override { return 0; }

  ///returns CLCT digis
  std::vector<CSCCLCTDigi> CLCTDigis(uint32_t idlayer) override;
  ///returns CorrelatedLCT digis
  std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis(uint32_t idlayer) const override;
  ///returns lct HMT Shower digi
  CSCShowerDigi showerDigi(uint32_t idlayer) const override { return CSCShowerDigi(); }
  ///returns anode HMT Shower digi
  CSCShowerDigi anodeShowerDigi(uint32_t idlayer) const override { return CSCShowerDigi(); }
  ///returns cathode HMT Shower digi
  CSCShowerDigi cathodeShowerDigi(uint32_t idlayer) const override { return CSCShowerDigi(); }

  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  unsigned short int sizeInWords() const override { return NWORDS; }

  unsigned short int NHeaderFrames() const override { return bits.nHeaderFrames; }
  /// returns the first data word
  unsigned short* data() override { return (unsigned short*)(&bits); }
  bool check() const override { return bits.e0bline == 0x6e0b && NHeaderFrames() + 1 == NWORDS; }

  /// for data packing
  void addCLCT0(const CSCCLCTDigi& digi) override;
  void addCLCT1(const CSCCLCTDigi& digi) override;
  void addALCT0(const CSCALCTDigi& digi) override;
  void addALCT1(const CSCALCTDigi& digi) override;
  void addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) override;
  void addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) override;
  void addShower(const CSCShowerDigi& digi) override {}
  void addAnodeShower(const CSCShowerDigi& digi) override {}
  void addCathodeShower(const CSCShowerDigi& digi) override {}

  void swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2);

  void print(std::ostream& os) const override;
  struct {
    unsigned b0cline : 16;
    unsigned nTBins : 5, dumpCFEBs : 7, fifoMode : 3, reserved_1 : 1;
    unsigned l1aNumber : 4, cscID : 4, boardID : 5, l1atype : 2, reserved_2 : 1;
    unsigned bxnCount : 12, r_type : 2, reserved_3 : 2;
    unsigned nHeaderFrames : 5, nCFEBs : 3, hasBuf : 1, preTrigTBins : 5, reserved_4 : 2;
    unsigned l1aTxCounter : 4, trigSourceVect : 8, hasPreTrig : 4;
    unsigned activeCFEBs : 5, CFEBsInstantiated : 5, runID : 4, reserved_6 : 2;
    unsigned bxnPreTrigger : 12, syncError : 1, reserved_7 : 3;

    unsigned clct0_valid : 1;
    unsigned clct0_quality : 3;
    unsigned clct0_shape : 3;
    unsigned clct0_strip_type : 1;
    unsigned clct0_bend : 1;
    unsigned clct0_key : 5;
    unsigned clct0_cfeb_low : 1;
    unsigned reserved_8 : 1;

    unsigned clct1_valid : 1;
    unsigned clct1_quality : 3;
    unsigned clct1_shape : 3;
    unsigned clct1_strip_type : 1;
    unsigned clct1_bend : 1;
    unsigned clct1_key : 5;
    unsigned clct1_cfeb_low : 1;
    unsigned reserved_9 : 1;

    unsigned clct0_cfeb_high : 2;
    unsigned clct0_bxn : 2;
    unsigned clct0_sync_err : 1;
    unsigned clct0_bx0_local : 1;
    unsigned clct1_cfeb_high : 2;
    unsigned clct1_bxn : 2;
    unsigned clct1_sync_err : 1;
    unsigned clct1_bx0_local : 1;
    unsigned invalidPattern : 1;
    unsigned reserved_10 : 3;

    unsigned tmbMatch : 1, alctOnly : 1, clctOnly : 1, bxn0Diff : 2, bxn1Diff : 2, alctMatchTime : 4, reserved_11 : 5;

    unsigned MPC_Muon0_wire_ : 7;
    unsigned MPC_Muon0_clct_pattern_ : 4;
    unsigned MPC_Muon0_quality_ : 4;
    unsigned reserved_12 : 1;

    unsigned MPC_Muon0_halfstrip_clct_pattern : 8;
    unsigned MPC_Muon0_bend_ : 1;
    unsigned MPC_Muon0_SyncErr_ : 1;
    unsigned MPC_Muon0_bx_ : 1;
    unsigned MPC_Muon0_bc0_ : 1;
    unsigned MPC_Muon0_cscid_low : 3;
    unsigned reserved_13 : 1;

    unsigned MPC_Muon1_wire_ : 7;
    unsigned MPC_Muon1_clct_pattern_ : 4;
    unsigned MPC_Muon1_quality_ : 4;
    unsigned reserved_14 : 1;

    unsigned MPC_Muon1_halfstrip_clct_pattern : 8;
    unsigned MPC_Muon1_bend_ : 1;
    unsigned MPC_Muon1_SyncErr_ : 1;
    unsigned MPC_Muon1_bx_ : 1;
    unsigned MPC_Muon1_bc0_ : 1;
    unsigned MPC_Muon1_cscid_low : 3;
    unsigned reserved_15 : 1;

    unsigned MPC_Muon0_vpf_ : 1;
    unsigned MPC_Muon0_cscid_bit4 : 1;
    unsigned MPC_Muon1_vpf_ : 1;
    unsigned MPC_Muon1_cscid_bit4 : 1;
    unsigned mpcAcceptLCT0 : 1;
    unsigned mpcAcceptLCT1 : 1;
    unsigned reserved_16_1 : 2;
    unsigned hs_thresh : 3;
    unsigned ds_thresh : 3;
    unsigned reserved_16_2 : 2;

    unsigned buffer_info_0 : 16;
    unsigned r_buf_nbusy : 4;
    unsigned buffer_info_1 : 12;
    unsigned buffer_info_2 : 16;
    unsigned buffer_info_3 : 16;
    unsigned alct_delay : 4, clct_width : 4, mpc_tx_delay : 4, reserved_21 : 4;

    unsigned rpc_exists : 2;
    unsigned rd_rpc_list : 2;
    unsigned rd_nrpcs : 2;
    unsigned rpc_read_enable : 1;
    unsigned r_nlayers_hit_vec : 3;
    unsigned pop_l1a_match_win : 4;
    unsigned reserved_22 : 2;

    unsigned bd_status : 14;
    unsigned reserved_23 : 2;
    unsigned uptime : 14;
    unsigned reserved_24 : 2;
    unsigned firmRevCode : 14, reserved_25 : 2;
    unsigned e0bline : 16;
  } bits;
};

#endif
