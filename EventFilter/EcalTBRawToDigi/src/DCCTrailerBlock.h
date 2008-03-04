// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef DCCTRAILERBLOCK_HH
#define DCCTRAILERBLOCK_HH


#include "DCCBlockPrototype.h"
class DCCDataParser;

class DCCTrailerBlock : public DCCBlockPrototype {

	public :
		
		DCCTrailerBlock(
			DCCDataParser * parser, 
			ulong * buffer, 
			ulong numbBytes,
			ulong wToEnd, 
			ulong wordEventOffset,
			ulong expectedLength,
			ulong expectedCRC
		);
		
		void dataCheck(); 
		
		
	protected :
		
		enum traillerFields{ EOE = 0xA};
		ulong expectedLength_;
		ulong expectedCRC_;


};

#endif

