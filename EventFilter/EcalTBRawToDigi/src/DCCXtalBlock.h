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
			ulong * buffer,
			ulong numbBytes,
			ulong wordsToEnd,  
			ulong wordEventOffset,
			ulong expectedXtalID ,
			ulong expectedStripID 
		);
		
		void dataCheck(); 
		int xtalID();
                                int stripID();
		std::vector<int> xtalDataSamples();

	protected :
		
		void increment(ulong numb);
		
		enum xtalBlockFields{ BPOSITION_BLOCKID = 30, BLOCKID = 3};
		
		ulong expectedXtalID_;
		ulong expectedStripID_;


};
#endif
