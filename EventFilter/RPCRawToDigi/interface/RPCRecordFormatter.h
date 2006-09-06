#ifndef RPCRecordFormatter_H
#define RPCRecordFormatter_H


/** \class Interprets the RPC record (16 bit) and fills the RPCDigiCollection
 *
 *  $Date: 2006/06/30 19:27:57 $
 *  $Revision: 1.10 $
 *  \author Ilaria Segoni - CERN
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <EventFilter/RPCRawToDigi/interface/RPCFEDData.h>

class RPCReadOutMapping;

class RPCRecordFormatter{
	public:
	   ///Creator 
	   RPCRecordFormatter(int fedId, const RPCReadOutMapping * readoutMapping);
	   
	   ///Destructor 
	   ~RPCRecordFormatter();
	   
	   /// Record Unpacker driver
	   /// Takes a reference to std::auto_ptr<RPCDigiCollection> because
	   /// I don't want to transfer ownership of RPCDigiCollection (I.S.)
	   void recordUnpack(RPCRecord & theRecord,std::auto_ptr<RPCDigiCollection> & prod, RPCFEDData & rawData, int triggerBX);
           
	   ///Unpack record type Start of BX Data and returns BXN
           int unpackBXRecord(const unsigned int* recordIndex); 
      
          ///Unpack record type TB Link Data (=> finds rmb and TB Link input number)
          void unpackTbLinkInputRecord(const unsigned int* recordIndex); 
      
          ///Unpack record type Link Board Data struct with LB payload
          RPCLinkBoardData  unpackLBRecord(const unsigned int* recordIndex); 
    
         ///Unpacks RMB corrupted/discarded data
          void unpackRMBCorruptedRecord(const unsigned int* recordIndex, 
	  	enum RPCRecord::recordTypes type, RPCFEDData & rawData);



      private:    
       int currentFED;
    	 int currentBX;
    	 int currentRMB;
    	 int currentTbLinkInputNumber;

  const RPCReadOutMapping * readoutMapping;
};

#endif
