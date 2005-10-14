#ifndef CSCTMBHeader_h
#define CSCTMBHeader_h

#include <iostream>
#include <iosfwd>

class CSCDMBHeader;

class CSCTMBHeader {

 public:
  CSCTMBHeader();
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
  short unsigned int Muon0_Frame0()    const {
    return MPC_Muon0_Frame0 | (MPC_Muon0_Frame0_bit15 <<15);}
  short unsigned int Muon0_Frame1()    const {
    return MPC_Muon0_Frame1 | (MPC_Muon0_Frame1_bit15 <<15);}
  short unsigned int Muon1_Frame0()    const {
    return MPC_Muon1_Frame0 | (MPC_Muon1_Frame0_bit15 <<15) ;}
  short unsigned int Muon1_Frame1()    const {
    return MPC_Muon1_Frame1 | (MPC_Muon1_Frame1_bit15 <<15) ;}
  short unsigned int ALCT_delay()   const {return alct_delay;}

  unsigned int clct0Word() const {return (CLCT0_low)|(CLCT0_high<<15);}
  unsigned int clct1Word() const {return (CLCT1_low)|(CLCT1_high<<15);}
  unsigned int clct0Word_low()  const {return (CLCT0_low) ;}
  unsigned int clct0Word_high() const {return (CLCT0_high);}

  unsigned int CLCT(const unsigned int index) const {
    if      (index == 0) return clct0Word();
    else if (index == 1) return clct1Word();
    else {
      std::cout << "+++ CSCTMBHeader:CLCT(): called with illegal index = "
	   << index << "! +++" << std::endl;
      return 0;
    }
  }

  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  unsigned short int sizeInWords() const     {return nHeaderFrames+1;}

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
  unsigned CLCT0_low:15, reserved_8:1;
  unsigned CLCT1_low:15, reserved_9:1;
  //line 10 (counting from 0)
  unsigned CLCT0_high:6, CLCT1_high:6, invalidPattern:1, reserved_10:3;
  unsigned tmbMatch:1, alctOnly:1, clctOnly:1, bxn0Diff:2, bxn1Diff:2,
           alctMatchTime:4, reserved_11:5;
  unsigned MPC_Muon0_Frame0:15, reserved_12:1;
  unsigned MPC_Muon0_Frame1:15, reserved_13:1;
  unsigned MPC_Muon1_Frame0:15, reserved_14:1;
  unsigned MPC_Muon1_Frame1:15, reserved_15:1;
  unsigned MPC_Muon0_Frame0_bit15:1, MPC_Muon0_Frame1_bit15:1,
           MPC_Muon1_Frame0_bit15:1, MPC_Muon1_Frame1_bit15:1,
           mpcAcceptLCT0:1, mpcAcceptLCT1:1, reserved_16:10;
  unsigned buffer_info_0:16;
  unsigned buffer_info_1:16;
  unsigned buffer_info_2:16;
  unsigned buffer_info_3:16;
  unsigned alct_delay:4,clct_width:4,mpc_tx_delay:4,reserved_21:4;
  unsigned buffer_info_5:16;
  unsigned buffer_info_6:16;
  unsigned buffer_info_7:16;
  unsigned firmRevCode:14, reserved_25:2;
  /// constant 6e0b
  unsigned e0bline:16;
  static bool debug;

};

#endif
