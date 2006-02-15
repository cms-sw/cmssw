#ifndef RPCRecordFormatter_H
#define RPCRecordFormatter_H


/** \class Interprets the RPC record (16 bit) and fills the RPCDigiCollection
 *
 *  $Date: 2006/02/14 10:51:21 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni - CERN
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <EventFilter/RPCRawToDigi/interface/RPCEventData.h>
#include <iostream>
#include <vector>


class RPCRecordFormatter{
	public:
	   ///Creator 
	   RPCRecordFormatter(bool printout);
	   
	   ///Destructor 
	   ~RPCRecordFormatter();
	   
	   /// Record Unpacker driver
	   /// Takes a reference to std::auto_ptr<RPCDigiCollection> because
	   /// I don't want to transfer ownership of RPCDigiCollection (I.S.)
	   void recordUnpack(RPCRecord::recordTypes typeOfRecord, const unsigned
	   			char* recordIndex, std::auto_ptr<RPCDigiCollection> & prod);
           
	   ///Unpack record type Start of BX Data and return BXN
           int unpackBXRecord(const unsigned char* recordIndex); 
      
          ///Unpack record type Channel Data and return DetId
          RPCDetId unpackChannelRecord(const unsigned char* recordIndex); 
      
          ///Unpack record type Chamber Data and return vector of Strip ID
          std::vector<int> unpackChamberRecord(const unsigned char* recordIndex); 
    
         ///Unpack RMB corrupted/discarded data
          void unpackRMBCorruptedRecord(const unsigned char* recordIndex);

         ///Returnss Container for DQM imformation
         RPCEventData eventData(){return rpcData;}


      private:    
    
         bool verbosity;  
         RPCEventData rpcData; 
};

#endif
