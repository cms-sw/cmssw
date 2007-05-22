#ifndef CSCTMBHeader_h
#define CSCTMBHeader_h

#include <iostream>
#include <iosfwd>
#include <vector>
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"

class CSCCLCTDigi;
class CSCDMBHeader;
class CSCCorrelatedLCTDigi;
class CSCTMBHeader {

 public:
  CSCTMBHeader();
  CSCTMBHeader(CSCTMBStatusDigi & digi)
    {
      memcpy(this, digi.header(), sizeInBytes()); 
    }

  uint16_t sizeInBytes() const {return 54;}

  /// fills fields like bxn and l1a
  void setEventInformation(const CSCDMBHeader &);
  short unsigned int FIFOMode()        const {return fifoMode;}
  short unsigned int DumpCFEBs()       const {return dumpCFEBs;}
  short unsigned int NTBins()          const {return nTBins;}
  short unsigned int BoardID()         const {return boardID;}
  short unsigned int CSCID()           const {return cscID;}
  short unsigned int BXNCount()        const {return bxnCount;}
  short unsigned int L1ANumber()       const {return l1aNumber;}
  short unsigned int PreTrigTBins()    const {return preTrigTBins;}
  short unsigned int NCFEBs()          const {return nCFEBs;}
  short unsigned int NHeaderFrames()   const {return nHeaderFrames;}
  short unsigned int TrigSourceVect()  const {return trigSourceVect;}
  short unsigned int ActiveCFEBs()     const {return activeCFEBs;}
  short unsigned int InstantiatedCFEBs() const {return CFEBsInstantiated;}
  short unsigned int BXNPreTrigger()   const {return bxnPreTrigger;}
  short unsigned int SyncError()       const {return syncError;}
  short unsigned int FirmRevCode()     const {return firmRevCode;}
//  short unsigned int WordCnt()         const {return wordCnt;}  // TMB header word cnt
//  short unsigned int CWordCnt()        const {return cWordCnt;} // "full" cathode word cnt
  short unsigned int Bxn1Diff()        const {return bxn1Diff;}
  short unsigned int Bxn0Diff()        const {return bxn0Diff;}
  short unsigned int CLCTOnly()        const {return clctOnly;}
  short unsigned int ALCTOnly()        const {return alctOnly;}
  short unsigned int TMBMatch()        const {return tmbMatch;}
  short unsigned int ALCTMatchTime()   const {return alctMatchTime;}
  short unsigned int MPCAcceptLCT0()   const {return mpcAcceptLCT0;}
  short unsigned int MPCAcceptLCT1()   const {return mpcAcceptLCT1;}
  short unsigned int HSThresh()        const {return hs_thresh;}
  short unsigned int DSThresh()        const {return ds_thresh;}


  short unsigned int RPCExists()       const {return rpc_exists;}
  short unsigned int RDRPCList()       const {return rd_rpc_list;}
  short unsigned int RDNRPCs()         const {return rd_nrpcs;}
  short unsigned int RPCReadEnable()   const {return rpc_read_enable;}
  short unsigned int NLayersHit()      const {return r_nlayers_hit_vec;}
  short unsigned int PopL1AMatch()     const {return pop_l1a_match_win;}

  
  short unsigned int MPC_Muon0_wire()          const {return MPC_Muon0_wire_;}
  short unsigned int MPC_Muon0_clct_pattern()  const {return MPC_Muon0_clct_pattern_;}
  short unsigned int MPC_Muon0_quality()       const {return MPC_Muon0_quality_;}
  short unsigned int MPC_Muon0_halfstrip_pat() const {return MPC_Muon0_halfstrip_clct_pattern;}
  short unsigned int MPC_Muon0_bend()          const {return MPC_Muon0_bend_;}
  short unsigned int MPC_Muon0_sync()          const {return MPC_Muon0_SyncErr_;}
  short unsigned int MPC_Muon0_bx()            const {return MPC_Muon0_bx_;}
  short unsigned int MPC_Muon0_bc0()           const {return MPC_Muon0_bc0_;}
  short unsigned int MPC_Muon0_cscid()         const {
    return MPC_Muon0_cscid_low | (MPC_Muon0_cscid_bit4<<3);}
  short unsigned int MPC_Muon0_valid()         const {return MPC_Muon0_vpf_;}
  
  short unsigned int MPC_Muon1_wire()          const {return MPC_Muon1_wire_;}
  short unsigned int MPC_Muon1_clct_pattern()  const {return MPC_Muon1_clct_pattern_;}
  short unsigned int MPC_Muon1_quality()       const {return MPC_Muon1_quality_;}
  short unsigned int MPC_Muon1_halfstrip_pat() const {return MPC_Muon1_halfstrip_clct_pattern;}
  short unsigned int MPC_Muon1_bend()          const {return MPC_Muon1_bend_;}
  short unsigned int MPC_Muon1_sync()          const {return MPC_Muon1_SyncErr_;}
  short unsigned int MPC_Muon1_bx()            const {return MPC_Muon1_bx_;}
  short unsigned int MPC_Muon1_bc0()           const {return MPC_Muon1_bc0_;}
  short unsigned int MPC_Muon1_cscid()         const {
    return MPC_Muon1_cscid_low | (MPC_Muon1_cscid_bit4<<3);}
  short unsigned int MPC_Muon1_valid()         const {return MPC_Muon1_vpf_;}
 


  short unsigned int ALCT_delay()   const {return alct_delay;}
  ///unsigned int clct0Word() const {return (CLCT0_low)|(CLCT0_high<<15);}
  ///unsigned int clct1Word() const {return (CLCT1_low)|(CLCT1_high<<15);}
  ///unsigned int clct0Word_low()  const {return (CLCT0_low) ;}
  ///unsigned int clct0Word_high() const {return (CLCT0_high);}
  ///these words are replaced by smaller constituents below:
  unsigned short int clct0Valid()     const {return clct0_valid;} 
  unsigned short int clct0Quality()   const {return clct0_quality;} 
  unsigned short int clct0Shape()     const {return clct0_shape;} 
  unsigned short int clct0StripType() const {return clct0_strip_type;} 
  unsigned short int clct0Bend()      const {return clct0_bend;} 
  unsigned short int clct0Key()       const {return clct0_key;} 
  unsigned short int clct0CFEB()      const {return (clct0_cfeb_low)|(clct0_cfeb_high<<1);} 
  unsigned short int clct0BXN()       const {return clct0_bxn;} 
  unsigned short int clct0SyncErr()   const {return clct0_sync_err;} 
  unsigned short int clct0BX0Local()  const {return clct0_bx0_local;} 

  unsigned short int clct1Valid()     const {return clct1_valid;} 
  unsigned short int clct1Quality()   const {return clct1_quality;} 
  unsigned short int clct1Shape()     const {return clct1_shape;} 
  unsigned short int clct1StripType() const {return clct1_strip_type;} 
  unsigned short int clct1Bend()      const {return clct1_bend;} 
  unsigned short int clct1Key()       const {return clct1_key;} 
  unsigned short int clct1CFEB()      const {return (clct1_cfeb_low)|(clct1_cfeb_high<<1);} 
  unsigned short int clct1BXN()       const {return clct1_bxn;} 
  unsigned short int clct1SyncErr()   const {return clct1_sync_err;} 
  unsigned short int clct1BX0Local()  const {return clct1_bx0_local;} 

 

  ///returns CLCT digis
  std::vector<CSCCLCTDigi> CLCTDigis() const;

  ///returns CorrelatedLCT digis
  std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis() const;

  ///these are broken into smaller words above
  /*unsigned int CLCT(const unsigned int index) const {
    if      (index == 0) return clct0Word();
    else if (index == 1) return clct1Word();
    else {
    //std::cout << "+++ CSCTMBHeader:CLCT(): called with illegal index = "
      //	   << index << "! +++" << std::endl;
      return 0;
      }
    }
  */

  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  unsigned short int sizeInWords() const     {return nHeaderFrames+1;}
  
  unsigned short * data() {return (unsigned short *) this;}

//  void SetCWordCnt(const unsigned short value) {cWordCnt = value;}

  /** turns on/off debug flag for this class */
  static void setDebug(const bool value) {debug = value;}

  bool check() const {return e0bline==0x6e0b;}

  friend std::ostream & operator<<(std::ostream & os, const CSCTMBHeader & hdr);
private:
  unsigned b0cline:16;
  unsigned nTBins:5, dumpCFEBs:7, fifoMode:3, reserved_1:1;
  unsigned l1aNumber:4, cscID:4, boardID:5, l1atype:2, reserved_2:1 ;
  unsigned bxnCount:12, r_type:2, reserved_3:2;
  unsigned nHeaderFrames:5, nCFEBs:3, hasBuf:1, preTrigTBins:5, reserved_4:2;
  unsigned l1aTxCounter:4, trigSourceVect:8, hasPreTrig:4;
  unsigned activeCFEBs:5, CFEBsInstantiated:5, runID:4, reserved_6:2;
  unsigned bxnPreTrigger:12, syncError:1, reserved_7:3;


  ///these words are broken into smaller pieces
  ///unsigned CLCT0_low:15, reserved_8:1;
  ///unsigned CLCT1_low:15, reserved_9:1;
  ///line 10 (counting from 0)
  ///unsigned CLCT0_high:6, CLCT1_high:6;
  
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
  /// constant 6e0b
  unsigned e0bline:16;
  static bool debug;

};

#endif
