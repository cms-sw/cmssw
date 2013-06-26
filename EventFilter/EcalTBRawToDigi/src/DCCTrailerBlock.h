// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef DCCTBTRAILERBLOCK_HH
#define DCCTBTRAILERBLOCK_HH


#include "DCCBlockPrototype.h"
class DCCDataParser;

class DCCTBTrailerBlock : public DCCTBBlockPrototype {

	public :
		
		DCCTBTrailerBlock(
			DCCTBDataParser * parser, 
			uint32_t * buffer, 
			uint32_t numbBytes,
			uint32_t wToEnd, 
			uint32_t wordEventOffset,
			uint32_t expectedLength,
			uint32_t expectedCRC
		);
		
		void dataCheck(); 
		
		
	protected :
		
		enum traillerFields{ EOE = 0xA};
		uint32_t expectedLength_;
		uint32_t expectedCRC_;


};

#endif

