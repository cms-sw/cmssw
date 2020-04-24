/*--------------------------------------------------------------*/
/* DCC TCC BLOCK CLASS                                          */
/*                                                              */
/* Author : N.Almeida (LIP)  Date   : 30/05/2005                */
/*--------------------------------------------------------------*/
#ifndef DCCTBTCCBLOCK_HH
#define DCCTBTCCBLOCK_HH

#include <iostream>                  //STL
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "DCCBlockPrototype.h"      //DATA DECODER
#include "DCCDataParser.h"
#include "DCCDataMapper.h"
#include "DCCEventBlock.h"

class DCCTBEventBlock;
class DCCTBDataParser;


class DCCTBTCCBlock : public DCCTBBlockPrototype {
	
public :
  /**
     Class constructor
  */
  DCCTBTCCBlock(DCCTBEventBlock * dccBlock,
	      DCCTBDataParser * parser, 
	      uint32_t * buffer, 
	      uint32_t numbBytes, 
	      uint32_t wordsToEnd,
	      uint32_t wordEventOffset,
	      uint32_t expectedId );     
  
  

  std::vector< std::pair<int, bool> > triggerSamples();
  
  std::vector<int> triggerFlags();
  
protected :
  /**
     Checks header's data
  */
  void dataCheck();
  
  /**
     Adds a new TCC block
  */
  using DCCTBBlockPrototype::increment;
  void  increment(uint32_t numb);
  
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
  
  DCCTBEventBlock * dccBlock_;
  uint32_t expectedId_;
};

#endif
