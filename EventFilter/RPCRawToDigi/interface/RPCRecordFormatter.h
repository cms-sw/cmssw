#ifndef RPCRecordFormatter_H
#define RPCRecordFormatter_H


/** \class Interprets the RPC record (16 bit) and fills the RPCDigiCollection
 *
 *  $Date: 2006/02/15 09:41:06 $
 *  $Revision: 1.4 $
 *  \author Ilaria Segoni - CERN
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <EventFilter/RPCRawToDigi/interface/RPCFEDData.h>
#include <iostream>
#include <vector>


class RPCRecordFormatter{
	public:
	   ///Creator 
	   RPCRecordFormatter();
	   
	   ///Destructor 
	   ~RPCRecordFormatter();
	   
	   /// Record Unpacker driver
	   /// Takes a reference to std::auto_ptr<RPCDigiCollection> because
	   /// I don't want to transfer ownership of RPCDigiCollection (I.S.)
	   void recordUnpack(RPCRecord & theRecord,std::auto_ptr<RPCDigiCollection> & prod, RPCFEDData & rawData);
           
	   ///Unpack record type Start of BX Data and return BXN
           int unpackBXRecord(const unsigned int* recordIndex); 
      
          ///Unpack record type Channel Data (=> finds rmb and channel number)
          void unpackChannelRecord(const unsigned int* recordIndex); 
      
          ///Unpack record type Link Board Data and return vector of BITS with
	  /// signal
          RPCLinkBoardData  unpackLBRecord(const unsigned int* recordIndex); 
    
         ///Unpacks RMB corrupted/discarded data
          void unpackRMBCorruptedRecord(const unsigned int* recordIndex, 
	  	enum RPCRecord::recordTypes type, RPCFEDData & rawData);



      private:    
    	 int currentRMB;
    	 int currentChannel;
};

#endif
