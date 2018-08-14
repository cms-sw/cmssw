#ifndef RPCRawToDigi_RPCPackingModule_H
#define RPCRawToDigi_RPCPackingModule_H

/** \class RPCPackingModule
 *  Driver class for digi to raw data conversions 
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "CondFormats/DataRecord/interface/RPCEMapRcd.h" 
#include "FWCore/Framework/interface/ESWatcher.h" 

#include <vector> 

namespace edm {class ParameterSet;}
namespace edm {class EventSetup; }
namespace edm {class Event; }

class FEDRawData;
class RPCRecordFormatter;
class RPCReadOutMapping;

class RPCPackingModule : public edm::stream::EDProducer<> {
public:

  /// ctor
  explicit RPCPackingModule( const edm::ParameterSet& );

  /// dtor
  ~RPCPackingModule() override;

  /// get data, convert to raw event, attach again to Event
  void produce( edm::Event&, const edm::EventSetup& ) override;

  static std::vector<rpcrawtodigi::EventRecords> eventRecords(
      int fedId, int trigger_BX, const RPCDigiCollection* , const RPCRecordFormatter& ); 

private:
  FEDRawData * rawData( int fedId, unsigned int lvl1_ID, const RPCDigiCollection* , const RPCRecordFormatter& ) const;

private:
  edm::EDGetTokenT<RPCDigiCollection> dataLabel_;
  edm::ESWatcher<RPCEMapRcd> recordWatcher_;
  const RPCReadOutMapping * theCabling; 
};
#endif
