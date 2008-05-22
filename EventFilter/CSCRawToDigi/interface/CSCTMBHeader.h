#ifndef CSCTMBHeader_h
#define CSCTMBHeader_h

///A.Tumanov Sept 18, 07

#include <iostream>
#include <iosfwd>
#include <vector>
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
class CSCDMBHeader;

struct CSCTMBHeader2006 {///this struct is for 2006 and earlier versions of dataformat
  CSCTMBHeader2006() {
    bzero(this, sizeInWords()*2);
  }
  
  short unsigned int sizeInWords() const {//size of TMBHeader
    return 27;
  }
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
};


struct CSCTMBHeader2007 {///this struct is for 2007 version of dataformat
  CSCTMBHeader2007() {
    bzero(this, sizeInWords()*2);
  }
  short unsigned int sizeInWords() const {//size of TMBHeader
    return 43;
  }
  void addALCT0(const CSCALCTDigi & digi);
  void addALCT1(const CSCALCTDigi & digi);

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
};


class CSCTMBHeader {

 public:
  CSCTMBHeader();
  CSCTMBHeader(const CSCTMBStatusDigi & digi);
  /// fills fields like bxn and l1a
  void setEventInformation(const CSCDMBHeader &);
  CSCTMBHeader(const unsigned short * buf);

  int FirmwareVersion() const {return firmwareVersion;}
  

  uint16_t BXNCount() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.bxnCount;
    case 2007:
      return header2007.bxnCount;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get bxncount: TMB firmware version is bad/not defined!";
      break;
    }
  }
  uint16_t ALCTMatchTime() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.alctMatchTime;
    case 2007:
      return header2007.matchWin;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get alctMatchTime: TMB firmware version is bad/not defined!";
      break;
    }
  }
  uint16_t CLCTOnly() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.clctOnly;
    case 2007:
      return header2007.clctOnly;
    default:
      edm::LogError("CSCTMBHeader")
	<<"coundn't get clctOnly: TMB firmware version is bad/not defined!";
      break;
    }
  }
  uint16_t ALCTOnly() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.alctOnly;
    case 2007:
      return header2007.alctOnly;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get alctOnly: TMB firmware version is bad/not defined!";
      break;
    }
  }
  uint16_t TMBMatch() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.tmbMatch;
    case 2007:
      return header2007.tmbMatch;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get TMBMatch : TMB firmware version is bad/not defined!";
      break;
    }
  }

  uint16_t Bxn0Diff() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.bxn0Diff;
    case 2007:
      return 0;///header2007.bxn0Diff;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get bxn0diff: TMB firmware version is bad/not defined!";
      break;
    }
  }
  uint16_t Bxn1Diff() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.bxn1Diff;
    case 2007:
      return 0;// header2007.bxn1Diff;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get bxn1diff: TMB firmware version is bad/not defined!";
      break;
    }
  }

  uint16_t L1ANumber() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.l1aNumber;
    case 2007:
      return header2007.l1aNumber;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get L1A: TMB firmware version is bad/not defined!";
      break;
    }
  }

  uint16_t sizeInBytes() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.sizeInWords()*2;
    case 2007:
      return header2007.sizeInWords()*2;
    default:
      edm::LogError("CSCTMBHeader")
	<<"coundn't get size: TMB firmware version is bad/not defined!";
      break;
    }
  }

  CSCTMBHeader2007 tmbHeader2007()   const {return header2007;}
  CSCTMBHeader2006 tmbHeader2006()   const {return header2006;}

  uint16_t NTBins() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.nTBins;
    case 2007:
      return header2007.nTBins;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get tbin: TMB firmware version is bad/not defined!";
      break;
    }
  }
  uint16_t NCFEBs() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.nCFEBs;
    case 2007:
      return header2007.nCFEBs;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't get ncfebs: TMB firmware version is bad/not defined!";
      break;
    }
  }


  ///returns CLCT digis
  std::vector<CSCCLCTDigi> CLCTDigis(uint32_t idlayer) const;
  ///returns CorrelatedLCT digis
  std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis(uint32_t idlayer) const;
 
  
  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  unsigned short int sizeInWords() const     {return NHeaderFrames()+1;}

  unsigned short int NHeaderFrames() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.nHeaderFrames;
    case 2007:
      return header2007.nHeaderFrames;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't header frames: TMB firmware version is bad/not defined!";
      break;
    }
  }
  
  unsigned short * data() {
    switch (firmwareVersion) {
    case 2006:
      memcpy(theOriginalBuffer, &header2006, header2006.sizeInWords()*2);
      break;
    case 2007:
      memcpy(theOriginalBuffer, &header2007, header2007.sizeInWords()*2);
      break;
    default:
      edm::LogError("CSCTMBHeader")
        <<"coundn't access data: TMB firmware version is bad/not defined!";
      break;
    }
    return theOriginalBuffer;
  }
  
  /** turns on/off debug flag for this class */
  static void setDebug(const bool value) {debug = value;}
  
  bool check() const {
    switch (firmwareVersion) {
    case 2006:
      return header2006.e0bline==0x6e0b;
    case 2007:
      return header2007.e0bline==0x6e0b;
    default:
      edm::LogError("CSCTMBHeader")
        <<"checked TMB firmware version to be  bad/not defined!";
      return false;
    }
  }

  /// for data packing
  void addCLCT0(const CSCCLCTDigi & digi);
  void addCLCT1(const CSCCLCTDigi & digi);
  void addALCT0(const CSCALCTDigi & digi);
  void addALCT1(const CSCALCTDigi & digi);
  void addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi);
  void addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi);

  // templated on the header struct
  template<typename T> void addCLCT0(const CSCCLCTDigi & digi, T & t);
  template<typename T> void addCLCT1(const CSCCLCTDigi & digi, T & t);
  template<typename T> void addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi, T & t);
  template<typename T> void addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi, T & t);

  /// tests that packing and unpacking give same results
  static void selfTest();
  
  friend std::ostream & operator<<(std::ostream & os, const CSCTMBHeader & hdr);

private:
  // helper method to reverse the strip numbers inthe ME1 readout
  void correctCLCTNumbering(uint32_t idlayer, int & strip, int & cfeb) const;
  void correctCorrLCTNumbering(uint32_t idlayer, int & strip) const;

  CSCTMBHeader2006 header2006;
  CSCTMBHeader2007 header2007;

  unsigned short int theOriginalBuffer[43];
  static bool debug;
  static unsigned short int firmwareVersion;
};


template<typename T> void
CSCTMBHeader::addCLCT0(const CSCCLCTDigi & digi, T & t) 
{
  //TODO correct strip and cfeb #
  t.clct0_valid = digi.isValid();
  t.clct0_quality = digi.getQuality();
  t.clct0_shape = digi.getPattern();
  t.clct0_bend = digi.getBend();
  t.clct0_key = digi.getStrip();
  t.clct0_cfeb_low = digi.getCFEB();
}

template<typename T> void
CSCTMBHeader::addCLCT1(const CSCCLCTDigi & digi, T & t)
{
  //TODO correct strip and cfeb #
  t.clct1_valid = digi.isValid();
  t.clct1_quality = digi.getQuality();
  t.clct1_shape = digi.getPattern();
  t.clct1_bend = digi.getBend();
  t.clct1_key = digi.getStrip();
  t.clct1_cfeb_low = digi.getCFEB();
}

template<typename T> void
CSCTMBHeader::addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi,  T & t)
{
  //TODO correct strips
  // Plus check where strip goes
  t.MPC_Muon0_wire_ = digi.getKeyWG();
  t.MPC_Muon0_clct_pattern_ = digi.getPattern();
  t.MPC_Muon0_quality_ = digi.getQuality();
  t.MPC_Muon0_halfstrip_clct_pattern = digi.getCLCTPattern();
  t.MPC_Muon0_bend_ = digi.getBend();
  t.MPC_Muon0_SyncErr_ = digi.getSyncErr();
  t.MPC_Muon0_bx_ = digi.getBX();
  t.MPC_Muon0_bc0_ = digi.getBX0();
  t.MPC_Muon0_cscid_low = digi.getCSCID();
}

template<typename T> void
CSCTMBHeader::addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi, T & t)
{
  //TODO correct strips
  // Plus check where strip goes
  t.MPC_Muon1_wire_ = digi.getKeyWG();
  t.MPC_Muon1_clct_pattern_ = digi.getPattern();
  t.MPC_Muon1_quality_ = digi.getQuality();
  t.MPC_Muon1_halfstrip_clct_pattern = digi.getCLCTPattern();
  t.MPC_Muon1_bend_ = digi.getBend();
  t.MPC_Muon1_SyncErr_ = digi.getSyncErr();
  t.MPC_Muon1_bx_ = digi.getBX();
  t.MPC_Muon1_bc0_ = digi.getBX0();
  t.MPC_Muon1_cscid_low = digi.getCSCID();
}


#endif
