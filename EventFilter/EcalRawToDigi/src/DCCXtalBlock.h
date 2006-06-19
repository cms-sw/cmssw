// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef DCCXTALBLOCK_HH
#define DCCXTALBLOCK_HH

#include "EventFilter/EcalRawToDigi/src/DCCBlockPrototype.h"
class DCCDataParser;

class DCCXtalBlock : public DCCBlockPrototype {

	public :
		
		DCCXtalBlock(
			DCCDataParser * parser, 
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
		vector<int> xtalDataSamples();
		
		
	protected :
		
		void increment(ulong numb);
		
		enum xtalBlockFields{ BPOSITION_BLOCKID = 30, BLOCKID = 3};
		
		ulong expectedXtalID_;
		ulong expectedStripID_;


};
#endif
