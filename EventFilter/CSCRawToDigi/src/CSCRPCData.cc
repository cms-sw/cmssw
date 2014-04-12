#include "EventFilter/CSCRawToDigi/interface/CSCRPCData.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <cstdio>
#include <strings.h> // for bzero
#include <cstring>

/** data format is
  RPC0  Tbin 0  Pads[7:0]
  RPC0  BXN     Pads[15:8]
  RPC0  Tbin 1  Pads[7:0]
...
  RPC0  Tbin 6  Pads[7:0]
  RPC3  BXN     Pads[15:8]
*/

bool CSCRPCData::debug = false;

CSCRPCData::CSCRPCData(int ntbins) 
  : ntbins_(ntbins), size_( 0 )
{
  theData[0] = 0x6b04;
  for(int i = 1; i < 257; ++i) {
    // data format is bits 12-14 are RPC number, 0 to 3
    int rpc = (i-1)/14;
    theData[i] = rpc << 12;

    // bits 8-11 of the first word of the pair is time bin
    int tbin = ((i-1)%14)/2;
    theData[i] |= tbin << 8;
  }
  theData[257] = 0x6e04;  
}

CSCRPCData::CSCRPCData(const unsigned short * buf, int length) 
  : size_(length)
{
  //size_ = ntbins_*2*4+2;
  // header & trailer word, + 4 RPCs per time bin, 2 lines per RPC
  ntbins_ = (size_-2)/8;
  memcpy(theData, buf, size_*2);
}

void CSCRPCData::Print() const {
  LogTrace ("CSCRPCData|CSCRawToDigi") << "CSCRPCData.Print";
  for(int line = 0; line < ((size_)); ++line) {
    LogTrace("CSCRPCData|CSCRawToDigi") <<std::hex << theData[line];
  }
  
  for(int linePair = 0; linePair < ((size_-2)/2); ++linePair) {
    // skip header word
    int pos = linePair*2 + 1;
    // make the two pad words into one and see if it's empty
    //int pad = theData[pos] & 0xff + ((theData[pos+1] & 0x3f) << 8);
  
    int bxnnew = ((theData[pos+1] >> 8)  & 0x7 );
  
    int rpc  = (theData[pos]   >> 12) & 0x7;
    int tbin = (theData[pos]   >> 8)  & 0xf;
    int bxn  = bxnnew;
  
    LogTrace ("CSCRPCData|CSCRawToDigi") << " RPC=" << rpc << " Tbin=" <<tbin <<" BXN=" << bxn;
  
  }
}

std::vector<int> CSCRPCData::BXN() const {
  std::vector<int> result;
  for(int linePair = 0; linePair < ((size_-2)/2); ++linePair) {
    // skip header word
    int pos = linePair*2 + 1;
    /// make the two pad words into one and see if it's empty
    //int pad = theData[pos] & 0xff + ((theData[pos+1] & 0x3f) << 8);
   
    int bxnnew = ((theData[pos+1] >> 8)  & 0x7 ) ;
    //int bxnnew = (((theData[pos+1] >> 8)  & 0x3 )<<2) | ((theData[pos+1]>>6)&0x3) ;
   
    int rpc  = (theData[pos]   >> 12) & 0x7;
    //int tbin = (theData[pos]   >> 8)  & 0xf;
    //int bxn  = bxnnew;
    result.push_back(bxnnew);
    result.push_back(rpc);

   
  }
  return result;
}

std::vector<CSCRPCDigi> CSCRPCData::digis() const {
  std::vector<CSCRPCDigi> result;
  int bxnold =0 ;
  int bxnnew =0 ;
  //int bxnewGreg;
  for(int linePair = 0; linePair < ((size_-2)/2); ++linePair) {
    // skip header word
    int pos = linePair*2 + 1;
    //  LogTrace("RPC") << "+++ CSCRPCData " << std::hex << theData[pos] 
	//			 << " " << theData[pos+1];
    if (debug) 
      LogTrace("CSCRPCData|CSCRawToDigi") << "+++ CSCRPCData " << std::hex << theData[pos] 
				 << " " << theData[pos+1];
    // make the two pad words into one and see if it's empty
    int pad = (theData[pos] & 0xff) + ((theData[pos+1] & 0xff) << 8);

    //bxnnew = (((theData[pos+1] >> 8)  & 0x3 )<<2) | ((theData[pos+1]>>6)&0x3) ;
    bxnnew = ((theData[pos+1] >> 8)  & 0x7 ) ;
    //LogTrace("RPC") << "               " << "bxnnew" << " " << bxnnew;
    //LogTrace("RPC") << "               " << "bxnnewGreg" << " " << bxnewGreg;
    if ( linePair == 0 ) bxnold = bxnnew;
    if ( bxnnew - bxnold > 1 ) 
      LogTrace("CSCRPCData|CSCRawToDigi") << "+++ CSCRPCData warning: RPC BXN is incrementing by more than 1 clock cycle";
    bxnold = bxnnew;

    if(pad != 0) {
      if (debug) LogTrace("CSCRPCData|CSCRawToDigi") << "+++ CSCRPCData Found a PAD =" 
					    << std::hex << pad << " " << theData[pos] 
					    << " + " << theData[pos+1];
      int rpc  = (theData[pos]   >> 12) & 0x7;
      int tbin = (theData[pos]   >> 8)  & 0xf;
      int bxn  = bxnnew;
      //LogTrace("RPC") << " rpc: " << rpc << " bxn: " << bxn << " tbin: " << tbin;
      for(int i = 0; i < 16; ++i) {
        // if the bit is set, make a digi
        if((pad>>i)&1) {
	  result.push_back(CSCRPCDigi(rpc, i, bxn, tbin));
	  //LogTrace("RPC") << "digi-->" << " rpc: " << rpc << " i: " << i << " bxn: " << bxn << " tbin: " << tbin;
        }
      }
    } 
  }
  return result;
}

