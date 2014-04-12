#ifndef RPCRecordFormatter_H
#define RPCRecordFormatter_H


/** \class Interprets the RPC raw data and fills the RPCDigiCollection
 */


#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCRawDataCounts.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"

class RPCReadOutMapping;
#include <vector>

class RPCRecordFormatter{
public:
  ///Creator 
  RPCRecordFormatter(int fedId, const RPCReadOutMapping * readoutMapping);
	   
  ///Destructor 
  ~RPCRecordFormatter();

  std::vector<rpcrawtodigi::EventRecords> recordPack(
      uint32_t rawDetId, const RPCDigi & digi, int trigger_BX) const; 

  int recordUnpack( const rpcrawtodigi::EventRecords & event, 
                    RPCDigiCollection * prod, 
                    RPCRawDataCounts * counter, 
                    RPCRawSynchro::ProdItem * synchro);

private:    
  int currentFED;
  int currentTbLinkInputNumber;

  const RPCReadOutMapping * readoutMapping;
};

#endif
