// Date   : 02/03/2004
// Author : N.Almeida (LIP)


#ifndef DCCTOWERBLOCK_HH
#define DCCTOWERBLOCK_HH

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>


#include "EventFilter/EcalRawToDigi/src/DCCBlockPrototype.h"

class DCCEventBlock;
class DCCXtalBlock;
class DCCDataParser;

class DCCTowerBlock : public DCCBlockPrototype {
	
	public :
		
		DCCTowerBlock(
			DCCEventBlock * dccBlock,
			DCCDataParser * parser, 
			ulong * buffer, 
			ulong numbBytes, 
			ulong wordsToEnd,
			ulong wordEventOffset,
			ulong expectedTowerID
		);
		
		~DCCTowerBlock();
		
		void parseXtalData();
                int towerID();
 		
		std::vector< DCCXtalBlock * > & xtalBlocks();
		
		std::vector< DCCXtalBlock * > xtalBlocksById(ulong stripId, ulong xtalId);
		
	protected :
		
		void dataCheck();
		
		enum towerFields{ BXMASK = 0xFFF,L1MASK = 0xFFF };
		
		std::vector<DCCXtalBlock * > xtalBlocks_;
		DCCEventBlock * dccBlock_;
		ulong expectedTowerID_;
		
		
};

inline std::vector<DCCXtalBlock *> & DCCTowerBlock::xtalBlocks(){ return xtalBlocks_; }

#endif
