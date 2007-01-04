#ifndef EventFilterRPCRawToDigiRPCRawDataPacker_H
#define EventFilterRPCRawToDigiRPCRawDataPacker_H

class FEDRawData;
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
class RPCRecordFormatter;


#include <boost/cstdint.hpp>

class RPCRawDataPacker {

public:
  FEDRawData * rawData(int fedId, const RPCDigiCollection* digis, const RPCRecordFormatter & formatter);
private:
  typedef uint64_t Word64;

private:
};

#endif
