// Date   : 30/05/2005
// Author : N.Almeida (LIP)


#ifndef DCCTCCBLOCK_HH
#define DCCTCCBLOCK_HH

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>


#include "EventFilter/EcalRawToDigi/src/DCCBlockPrototype.h"

class DCCEventBlock;
class DCCDataParser;

class DCCTCCBlock : public DCCBlockPrototype {
	
	public :
		
		DCCTCCBlock(
			DCCEventBlock * dccBlock,
			DCCDataParser * parser, 
			ulong * buffer, 
			ulong numbBytes, 
			ulong wordsToEnd,
			ulong wordEventOffset,
			ulong expectedId
		);
	
		
		
	protected :
		
		void dataCheck();
		
		void  increment(ulong numb);
		
		enum tccFields{ 
			BXMASK = 0xFFF,
			L1MASK = 0xFFF, 
			BPOSITION_BLOCKID = 29,
			BLOCKID = 3
		};
	
		DCCEventBlock * dccBlock_;
		
		ulong expectedId_;
		
		
};

#endif
