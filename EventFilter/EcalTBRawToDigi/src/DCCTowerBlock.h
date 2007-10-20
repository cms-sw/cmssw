// Date   : 02/03/2004
// Author : N.Almeida (LIP)


#ifndef DCCTBTOWERBLOCK_HH
#define DCCTBTOWERBLOCK_HH

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>


#include "DCCBlockPrototype.h"

class DCCTBEventBlock;
class DCCTBXtalBlock;
class DCCTBDataParser;

class DCCTBTowerBlock : public DCCTBBlockPrototype {
	
	public :
		
		DCCTBTowerBlock(
			DCCTBEventBlock * dccBlock,
			DCCTBDataParser * parser, 
			ulong * buffer, 
			ulong numbBytes, 
			ulong wordsToEnd,
			ulong wordEventOffset,
			ulong expectedTowerID
		);
		
		~DCCTBTowerBlock();
		
		void parseXtalData();
		int towerID();

		std::vector< DCCTBXtalBlock * > & xtalBlocks();
		
		std::vector< DCCTBXtalBlock * > xtalBlocksById(ulong stripId, ulong xtalId);
		
	protected :
		
		void dataCheck();
		
		enum towerFields{ BXMASK = 0xFFF,L1MASK = 0xFFF };
		
		std::vector<DCCTBXtalBlock * > xtalBlocks_;
		DCCTBEventBlock * dccBlock_;
		ulong expectedTowerID_;
		
		
};

inline std::vector<DCCTBXtalBlock *> & DCCTBTowerBlock::xtalBlocks(){ return xtalBlocks_; }

#endif
