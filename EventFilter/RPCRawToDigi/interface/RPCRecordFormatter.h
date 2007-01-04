#ifndef RPCRecordFormatter_H
#define RPCRecordFormatter_H


/** \class Interprets the RPC record (16 bit) and fills the RPCDigiCollection
 *
 *  $Date: 2006/10/08 12:08:50 $
 *  $Revision: 1.12 $
 *  \author Ilaria Segoni - CERN
 */


#include "EventFilter/RPCRawToDigi/interface/RPCRecord.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCFEDData.h"

class RPCReadOutMapping;
class FEDRawData;

class RPCRecordFormatter{
public:
  typedef uint16_t Record;
	   ///Creator 
	   RPCRecordFormatter(int fedId, const RPCReadOutMapping * readoutMapping);
	   
	   ///Destructor 
	   ~RPCRecordFormatter();

         int pack( uint32_t rawDetId, const RPCDigi & digi, int trigger_BX, 
             Record & bxRecord, Record & tbRecord, Record & lbRecord) const;
         
         static void setEmptyRecord(Record& record);
         static void setBXRecord( Record& record, int bx);
         static void setTBRecord( Record& record, int tbLinkInputNumber, int rmb);
         static void setLBRecord( Record& record, const RPCLinkBoardData & lbData);
	   
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

         ///Unpacks RMB disabled ID
          void unpackRMBDisabledRecord(const unsigned int* recordIndex, 
	  	enum RPCRecord::recordTypes type, RPCFEDData & rawData);



      private:    
       int currentFED;
    	 int currentBX;
    	 int currentRMB;
    	 int currentTbLinkInputNumber;

  const RPCReadOutMapping * readoutMapping;
};

#endif
