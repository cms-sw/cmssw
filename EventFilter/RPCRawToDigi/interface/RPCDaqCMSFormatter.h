#ifndef RPCDaqCMSFormatter_H
#define RPCDaqCMSFormatter_H
/*
      
      
 */

#include <DataFormats/RPCDigis/interface/RPCDigiCollection.h>
#include <string>

namespace raw {class FEDRawData;}


class RPCDaqCMSFormatter {

public:
  
	RPCDaqCMSFormatter();
	~RPCDaqCMSFormatter();

	void interpretRawData( const raw::FEDRawData & data, 
	        	      RPCDigiCollection& digicollection );

	int headerUnpack( RPCDigiCollection & digicollection );		      
	void payLoadUnpack(int payLoadSize , RPCDigiCollection & digicollection);		      
	void TrailerUnpack( RPCDigiCollection & digicollection );		      

	inline void checkMemory(int totsize, int alreadyfilled, int requested){
	  if (alreadyfilled+requested > totsize) throw std::string("DaqFEDFormatter::checkMemory() - ERROR: the requested memory exceeds the reserved one");

	}

private:
	
	int length;  
	int bytesread;  


};

#endif
