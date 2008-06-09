#ifndef RPCRecordFormatter_H
#define RPCRecordFormatter_H


/** \class Interprets the RPC record (16 bit) and fills the RPCDigiCollection
 *
 *  $Date: 2007/01/04 21:29:03 $
 *  $Revision: 1.13 $
 *  \author Ilaria Segoni - CERN
 */


#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
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

  void recordUnpack(const rpcrawtodigi::EventRecords & event, 
                    std::auto_ptr<RPCDigiCollection> & prod);

private:    
  int currentFED;
  int currentTbLinkInputNumber;

  const RPCReadOutMapping * readoutMapping;
};

#endif
