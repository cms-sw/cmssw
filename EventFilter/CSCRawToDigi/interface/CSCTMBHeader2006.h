#ifndef CSCTMBHeader2006_h
#define CSCTMBHeader2006_h
#include "EventFilter/CSCRawToDigi/interface/CSCVTMBHeaderFormat.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"


struct CSCTMBHeader2006 : public CSCVTMBHeaderFormat {
  enum {NWORDS=27};
  CSCTMBHeader2006();
  explicit CSCTMBHeader2006(const unsigned short * buf);
  virtual void setEventInformation(const CSCDMBHeader & dmbHeader);

  virtual uint16_t BXNCount() const {return bits.bxnCount;}
  virtual uint16_t ALCTMatchTime() const {return bits.alctMatchTime;}
  virtual uint16_t CLCTOnly() const {return bits.clctOnly;}
  virtual uint16_t ALCTOnly() const {return bits.alctOnly;}
  virtual uint16_t TMBMatch() const {return bits.tmbMatch;}
  virtual uint16_t Bxn0Diff() const {return bits.bxn0Diff;}
  virtual uint16_t Bxn1Diff() const {return bits.bxn1Diff;}
  virtual uint16_t L1ANumber() const {return bits.l1aNumber;}
  virtual uint16_t NTBins() const {return bits.nTBins;}
  virtual uint16_t NCFEBs() const {return bits.nCFEBs;}
  virtual void setNCFEBs(uint16_t ncfebs) {bits.nCFEBs = ncfebs & 0x1F;}
  virtual uint16_t firmwareRevision() const {return bits.firmRevCode;}
  virtual uint16_t syncError() const {return bits.syncError;}
  virtual uint16_t syncErrorCLCT() const {return (bits.clct0_sync_err | bits.clct1_sync_err);}
  virtual uint16_t syncErrorMPC0() const {return bits.MPC_Muon0_SyncErr_;}
  virtual uint16_t syncErrorMPC1() const {return bits.MPC_Muon1_SyncErr_;}

  ///returns CLCT digis
  virtual std::vector<CSCCLCTDigi> CLCTDigis(uint32_t idlayer);
  ///returns CorrelatedLCT digis
  virtual std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis(uint32_t idlayer) const;
 
  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  unsigned short int sizeInWords() const {return NWORDS;}

  virtual unsigned short int NHeaderFrames() const {return bits.nHeaderFrames;}
  /// returns the first data word
  virtual unsigned short * data() {return (unsigned short *)(&bits);}
  virtual bool check() const {return bits.e0bline==0x6e0b && NHeaderFrames()+1 == NWORDS;}

  /// for data packing
  virtual void addCLCT0(const CSCCLCTDigi & digi);
  virtual void addCLCT1(const CSCCLCTDigi & digi);
  virtual void addALCT0(const CSCALCTDigi & digi);
  virtual void addALCT1(const CSCALCTDigi & digi);
  virtual void addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi);
  virtual void addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi);

  void swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2);

  virtual void print(std::ostream & os) const;
    struct {
      unsigned b0cline:16;
      unsigned nTBins:5, dumpCFEBs:7, fifoMode:3, reserved_1:1;
      unsigned l1aNumber:4, cscID:4, boardID:5, l1atype:2, reserved_2:1 ;
      unsigned bxnCount:12, r_type:2, reserved_3:2;
      unsigned nHeaderFrames:5, nCFEBs:3, hasBuf:1, preTrigTBins:5, reserved_4:2;
      unsigned l1aTxCounter:4, trigSourceVect:8, hasPreTrig:4;
      unsigned activeCFEBs:5, CFEBsInstantiated:5, runID:4, reserved_6:2;
      unsigned bxnPreTrigger:12, syncError:1, reserved_7:3;

      unsigned clct0_valid      :1;
      unsigned clct0_quality    :3;
      unsigned clct0_shape      :3;
      unsigned clct0_strip_type :1;
      unsigned clct0_bend       :1;
      unsigned clct0_key        :5;
      unsigned clct0_cfeb_low   :1;
      unsigned reserved_8       :1;

      unsigned clct1_valid      :1;
      unsigned clct1_quality    :3;
      unsigned clct1_shape      :3;
      unsigned clct1_strip_type :1;
      unsigned clct1_bend       :1;
      unsigned clct1_key        :5;
      unsigned clct1_cfeb_low   :1;
      unsigned reserved_9       :1;

      unsigned clct0_cfeb_high  :2;
      unsigned clct0_bxn        :2;
      unsigned clct0_sync_err   :1;
      unsigned clct0_bx0_local  :1;
      unsigned clct1_cfeb_high  :2;
      unsigned clct1_bxn        :2;
      unsigned clct1_sync_err   :1;
      unsigned clct1_bx0_local  :1;
      unsigned invalidPattern   :1;
      unsigned reserved_10      :3;

      unsigned tmbMatch:1, alctOnly:1, clctOnly:1, bxn0Diff:2, bxn1Diff:2,
               alctMatchTime:4, reserved_11:5;

      unsigned MPC_Muon0_wire_         : 7;
      unsigned MPC_Muon0_clct_pattern_ : 4;
      unsigned MPC_Muon0_quality_      : 4;
      unsigned reserved_12:1;

      unsigned MPC_Muon0_halfstrip_clct_pattern : 8;
      unsigned MPC_Muon0_bend_                  : 1;
      unsigned MPC_Muon0_SyncErr_               : 1;
      unsigned MPC_Muon0_bx_                    : 1;
      unsigned MPC_Muon0_bc0_                   : 1;
      unsigned MPC_Muon0_cscid_low              : 3;
      unsigned reserved_13:1;

      unsigned MPC_Muon1_wire_         : 7;
      unsigned MPC_Muon1_clct_pattern_ : 4;
      unsigned MPC_Muon1_quality_      : 4;
      unsigned reserved_14:1;

      unsigned MPC_Muon1_halfstrip_clct_pattern : 8;
      unsigned MPC_Muon1_bend_                  : 1;
      unsigned MPC_Muon1_SyncErr_               : 1;
      unsigned MPC_Muon1_bx_                    : 1;
      unsigned MPC_Muon1_bc0_                   : 1;
      unsigned MPC_Muon1_cscid_low              : 3;
      unsigned reserved_15:1;

      unsigned MPC_Muon0_vpf_        : 1;
      unsigned MPC_Muon0_cscid_bit4  : 1;
      unsigned MPC_Muon1_vpf_        : 1;
      unsigned MPC_Muon1_cscid_bit4  : 1;
      unsigned mpcAcceptLCT0         : 1;
      unsigned mpcAcceptLCT1         : 1;
      unsigned reserved_16_1         : 2;
      unsigned hs_thresh             : 3;
      unsigned ds_thresh             : 3;
      unsigned reserved_16_2:2;

      unsigned buffer_info_0:16;
      unsigned r_buf_nbusy:4; unsigned buffer_info_1:12;
      unsigned buffer_info_2:16;
      unsigned buffer_info_3:16;
      unsigned alct_delay:4,clct_width:4,mpc_tx_delay:4,reserved_21:4;

      unsigned rpc_exists:2;
      unsigned rd_rpc_list:2;
      unsigned rd_nrpcs:2;
      unsigned rpc_read_enable:1;
      unsigned r_nlayers_hit_vec:3;
      unsigned pop_l1a_match_win:4;
      unsigned reserved_22:2;

      unsigned bd_status :14;  unsigned reserved_23:2;
      unsigned uptime :14;  unsigned reserved_24:2;
      unsigned firmRevCode:14, reserved_25:2;
      unsigned e0bline:16;
    } bits;

};

#endif

