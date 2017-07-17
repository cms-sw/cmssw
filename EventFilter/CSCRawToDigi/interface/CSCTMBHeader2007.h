#ifndef CSCTMBHeader2007_h
#define CSCTMBHeader2007_h
#include "EventFilter/CSCRawToDigi/interface/CSCVTMBHeaderFormat.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"


struct CSCTMBHeader2007 : public CSCVTMBHeaderFormat {
  enum {NWORDS = 43};
  CSCTMBHeader2007();
  CSCTMBHeader2007(const unsigned short * buf);
  virtual void setEventInformation(const CSCDMBHeader & dmbHeader);

  virtual uint16_t BXNCount() const {return bits.bxnCount;}
  virtual uint16_t ALCTMatchTime() const {return bits.matchWin;}
  virtual uint16_t CLCTOnly() const {return bits.clctOnly;}
  virtual uint16_t ALCTOnly() const {return bits.alctOnly;}
  virtual uint16_t TMBMatch() const {return bits.tmbMatch;}
  virtual uint16_t Bxn0Diff() const {return 0;}
  virtual uint16_t Bxn1Diff() const {return 0;}
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
  unsigned short int sizeInWords() const     {return NWORDS;}

  virtual unsigned short int NHeaderFrames() const {return bits.nHeaderFrames;}
  /// returns the first data word
  virtual unsigned short * data() {return (unsigned short *)(&bits);}
  virtual bool check() const {return bits.e0bline==0x6e0b;}

  /// Needed before data packing
  //void setChamberId(const CSCDetId & detId) {theChamberId = detId;}

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
    unsigned bxnCount:12, dduCode1:3, flag1:1;
    unsigned l1aNumber:12, dduCode2:3, flag2:1;
    unsigned readoutCounter:12, dduCode3:3, flag3:1;
    unsigned boardID:5, cscID:4, runID:4, stackOvf:1, syncError:1, flag4:1;
    unsigned nHeaderFrames:6, fifoMode:3, r_type:2, l1atype:2, hasBuf:1, bufFull:1, flag5:1;
    unsigned bd_status:15, flag6:1;
    unsigned firmRevCode:15, flag7:1;
    unsigned bxnPreTrigger:12, reserved:3, flag8:1;
    unsigned preTrigCounterLow:15, flag9:1;
    unsigned preTrigCounterHigh:15, flag10:1;
    unsigned clctCounterLow:15, flag11:1;
    unsigned clctCounterHigh:15, flag12:1;
    unsigned trigCounterLow:15, flag13:1;
    unsigned trigCounterHigh:15, flag14:1;
    unsigned alctCounterLow:15, flag15:1;
    unsigned alctCounterHigh:15, flag16:1;
    unsigned uptimeCounterLow:15, flag17:1;
    unsigned uptimeCounterHigh:15, flag18:1;
    unsigned nCFEBs:3, nTBins:5, fifoPretrig:5, scopeExists:1, vmeExists:1, flag19:1;
    unsigned hitThresh:3, pidThresh:4, nphThresh:3, lyrThresh:3, layerTrigEnabled:1, staggerCSC:1, flag20:1;
    unsigned triadPersist:4, dmbThresh:3, alct_delay:4, clct_width:4, flag21:1;
    unsigned trigSourceVect:9, r_nlayers_hit_vec:6, flag22:1;
    unsigned activeCFEBs:5, readCFEBs:5, pop_l1a_match_win:4, layerTriggered:1, flag23:1;
    unsigned tmbMatch:1, alctOnly:1, clctOnly:1, matchWin:4, noTMBTrig:1, noMPCFrame:1, noMPCResponse:1, reserved1:5, flag24:1;
    unsigned clct0_valid:1, clct0_quality:3, clct0_shape:4, clct0_bend:1, clct0_key:5, clct0_cfeb_low:1, flag25:1;
    unsigned clct1_valid:1, clct1_quality:3, clct1_shape:4, clct1_bend:1, clct1_key:5, clct1_cfeb_low:1, flag26:1;
    unsigned clct0_cfeb_high:2, clct0_bxn:2, clct0_sync_err:1, clct0_bx0_local:1, clct1_cfeb_high:2, clct1_bxn:2, clct1_sync_err:1, clct1_bx0_local:1, clct0Invalid:1, clct1Invalid:1, clct1Busy:1, flag27:1;
    unsigned alct0Valid:1, alct0Quality:2, alct0Amu:1, alct0Key:7, reserved2:4, flag28:1;
    unsigned alct1Valid:1, alct1Quality:2, alct1Amu:1, alct1Key:7, reserved3:4, flag29:1;
    unsigned alctBXN:5, alctSeqStatus:2, alctSEUStatus:2, alctReserved:4, alctCfg:1, reserved4:1, flag30:1;
    unsigned MPC_Muon0_wire_:7, MPC_Muon0_clct_pattern_:4, MPC_Muon0_quality_:4, flag31:1;
    unsigned MPC_Muon0_halfstrip_clct_pattern:8, MPC_Muon0_bend_:1, MPC_Muon0_SyncErr_:1, MPC_Muon0_bx_:1, MPC_Muon0_bc0_:1, MPC_Muon0_cscid_low:3, flag32:1;
    unsigned MPC_Muon1_wire_: 7, MPC_Muon1_clct_pattern_:4, MPC_Muon1_quality_:4, flag33:1;
    unsigned MPC_Muon1_halfstrip_clct_pattern:8, MPC_Muon1_bend_:1, MPC_Muon1_SyncErr_:1, MPC_Muon1_bx_:1, MPC_Muon1_bc0_:1, MPC_Muon1_cscid_low:3, flag34:1;
    unsigned MPC_Muon0_vpf_:1, MPC_Muon0_cscid_bit4:1, MPC_Muon1_vpf_:1, MPC_Muon1_cscid_bit4:1, MPCDelay:4, MPCAccept:2, CFEBsEnabled:5, flag35:1;
    unsigned RPCExists:2, RPCList:2, NRPCs:2, RPCEnable:1, RPCMatch:8, flag36:1;
    unsigned addrPretrig:12, bufReady:1, reserved5:2, flag37:1;
    unsigned addrL1a:12, reserved6:3, flag38:1;
    unsigned reserved7:15, flag39:1;
    unsigned reserved8:15, flag40:1;
    unsigned reserved9:15, flag41:1;
    unsigned e0bline:16;
  } bits;


};

#endif

