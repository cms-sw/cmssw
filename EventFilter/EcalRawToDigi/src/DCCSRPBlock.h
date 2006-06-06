// Date   : 30/05/2005
// Author : N.Almeida (LIP)


#ifndef DCCSRPBLOCK_HH
#define DCCSRPBLOCK_HH

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>


#include "DCCBlockPrototype.h"

class DCCEventBlock;
class DCCXtalBlock;
class DCCDataParser;

class DCCSRPBlock : public DCCBlockPrototype {
	
	public :
		
		DCCSRPBlock(
			DCCEventBlock * dccBlock,
			DCCDataParser * parser, 
			ulong * buffer, 
			ulong numbBytes,
			ulong wordsToEnd, 
			ulong wordEventOffset
		);
	
		
		
	protected :
		
		void dataCheck();
		
		void  increment(ulong numb);
		
		enum srpFields{ 
			BXMASK = 0xFFF,
			L1MASK = 0xFFF, 
			BPOSITION_BLOCKID = 29,
			BLOCKID = 4
		};
	
		DCCEventBlock * dccBlock_;
		
		
		
};

#endif
