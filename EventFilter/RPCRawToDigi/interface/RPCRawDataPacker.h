#ifndef EventFilterRPCRawToDigiRPCRawDataPacker_H
#define EventFilterRPCRawToDigiRPCRawDataPacker_H

class FEDRawData;
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"

#include <boost/cstdint.hpp>
#include <vector>

////////////////
#include "IORawData/RPCFileReader/interface/RPCPacData.h"
#include "IORawData/RPCFileReader/interface/LinkDataXMLWriter.h"
////////////////


class RPCRawDataPacker {

public:
  ///////////////////////////
  RPCRawDataPacker();
  ~RPCRawDataPacker();
  ///////////////////////////

  FEDRawData * rawData(
      int fedId, const RPCDigiCollection* digis, const RPCRecordFormatter & formatter);

private:
  typedef uint64_t Word64;

  typedef RPCRecordFormatter::Record Record;
  struct Records { 
    Record bx; Record tb; Record lb;  
    bool samePartition(const Records & r) const;
  };

  std::vector<Records> margeRecords(const std::vector<Records> & data) const;

  /////////////////////////////////////////////////////////////
  LinkDataXMLWriter *myXMLWriter;
  std::vector<std::vector<RPCPacData> > linkData_;//(nBX,18)
  /////////////////////////////////////////////////////////////


private:
};

#endif
