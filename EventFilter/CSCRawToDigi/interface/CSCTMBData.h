//_______________________________________
//
//  Class to group TMB data  
//  CSCTMBData 9/18/03  B.Mohr           
//_______________________________________
//

#ifndef CSCTMBData_h
#define CSCTMBData_h

#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBScope.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCRPCData.h"
#include <bitset>
#include <boost/dynamic_bitset.hpp>


class CSCTMBData {

 public:

  CSCTMBData();
  ~CSCTMBData();
  CSCTMBData(unsigned short *buf);
  int UnpackTMB(unsigned short *buf);
  /// sees if the size adds up to the word count
  bool checkSize() const;
  static void setDebug(const bool value) {debug = value;}
  short unsigned int   CWordCnt()   const {return cWordCnt;}
  int getCRC() const {return CRCCnt;}
  const unsigned short size()       const {return size_;}

  CSCTMBHeader & tmbHeader()   {return theTMBHeader;}
  CSCCLCTData & clctData()     {return theCLCTData;}
  /// check this before using TMB Scope
  bool hasTMBScope() const { return theTMBScopeIsPresent;}
  CSCTMBScope & tmbScope() const;
  CSCTMBTrailer & tmbTrailer() {return theTMBTrailer;}
  /// check this before using RPC
  bool hasRPC() const {return theRPCDataIsPresent;}
  CSCRPCData & rpcData()       {return theRPCData;}

  /// not const because it sets size int TMBTrailer
  boost::dynamic_bitset<> pack();
  std::bitset<22> calCRC22(const std::vector< std::bitset<16> >& datain);
  std::bitset<22> nextCRC22_D16(const std::bitset<16>& D, const std::bitset<22>& C);
  int TMBCRCcalc();
  
 private:

  int CRCCnt;
  ///@@ not sure what this means for simulation.  I keep this
  /// around so we can calculate CRCs
  unsigned short * theOriginalBuffer;
  /// CRC calc needs to know where 0x6B0C and 0x6E0F lines were
  /// we want to put off CRC calc until needed
  unsigned theB0CLine;
  unsigned theE0FLine;

  CSCTMBHeader theTMBHeader;
  CSCCLCTData theCLCTData;
  CSCRPCData theRPCData;
  /// The tmb scope is not present in most of data hence its dynamic
  bool theTMBScopeIsPresent;
  CSCTMBScope * theTMBScope;

  CSCTMBTrailer theTMBTrailer;
  static bool debug;
  unsigned short size_;
  unsigned short cWordCnt;
  bool theRPCDataIsPresent;
};

#endif
