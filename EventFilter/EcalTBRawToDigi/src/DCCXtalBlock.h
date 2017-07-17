// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef DCCTBXTALBLOCK_HH
#define DCCTBXTALBLOCK_HH

#include "DCCBlockPrototype.h"
class DCCTBDataParser;

class DCCTBXtalBlock : public DCCTBBlockPrototype {

	public :
		
		DCCTBXtalBlock(
			DCCTBDataParser * parser, 
			uint32_t * buffer,
			uint32_t numbBytes,
			uint32_t wordsToEnd,  
			uint32_t wordEventOffset,
			uint32_t expectedXtalID ,
			uint32_t expectedStripID 
		);
		
		void dataCheck(); 
		int xtalID();
                                int stripID();
		std::vector<int> xtalDataSamples();

	protected :
		using DCCTBBlockPrototype::increment;
		void increment(uint32_t numb);
		
		enum xtalBlockFields{ BPOSITION_BLOCKID = 30, BLOCKID = 3};
		
		uint32_t expectedXtalID_;
		uint32_t expectedStripID_;


};
#endif
