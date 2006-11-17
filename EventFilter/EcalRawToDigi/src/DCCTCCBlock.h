/*--------------------------------------------------------------*/
/* DCC TCC BLOCK CLASS                                          */
/*                                                              */
/* Author : N.Almeida (LIP)  Date   : 30/05/2005                */
/*--------------------------------------------------------------*/
#ifndef DCCTCCBLOCK_HH
#define DCCTCCBLOCK_HH

#include <iostream>                  //STL
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "DCCBlockPrototype.h"      //DATA DECODER
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
#include "DCCEventBlock.h"

class DCCEventBlock;
class DCCDataParser;


class DCCTCCBlock : public DCCBlockPrototype {
	
public :
  /**
     Class constructor
  */
  DCCTCCBlock(DCCEventBlock * dccBlock,
	      DCCDataParser * parser, 
	      ulong * buffer, 
	      ulong numbBytes, 
	      ulong wordsToEnd,
	      ulong wordEventOffset,
	      ulong expectedId );     
  
  

  vector< pair<int, bool> > triggerSamples();
  
  vector<int> triggerFlags();
  
protected :
  /**
     Checks header's data
  */
  void dataCheck();
  
  /**
     Adds a new TCC block
  */
  void  increment(ulong numb);
  
  /**
     Define TCC block fields
     BXMASK (mask for BX, 12bit)
     L1MASK (mask for LV1, 12 bit)
     BPOSITION_BLOCKID (bit position for TCC block id, bit=61/29 for 64/32bit words
     BLOCKID (TCC block id B'011' = 3)
  */
  enum tccFields{ 
    BXMASK = 0xFFF,
    L1MASK = 0xFFF, 
    BPOSITION_BLOCKID = 29,
    BLOCKID = 3,
    BPOSITION_FGVB = 8,
    ETMASK = 0xFF                  
  };
  
  DCCEventBlock * dccBlock_;
  ulong expectedId_;
};

#endif
