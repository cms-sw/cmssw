#ifndef EventFilter_RPCRawToDigi_DataRecord_H
#define EventFilter_RPCRawToDigi_DataRecord_H

#include <boost/cstdint.hpp>
#include <string>
#include <bitset>
#include <sstream>

namespace rpcrawtodigi { 
class DataRecord {
public:
  typedef uint16_t RecordType;
  enum recordName {
    StartOfBXData = 1,
    StartOfTbLinkInputNumberData = 2,
    LinkBoardData = 3,
    EmptyWord     = 4,
    RMBDiscarded  = 5,
    RMBCorrupted  = 6,
    DCCDiscarded  = 7,
    RMBDisabled   = 8,
    UndefinedType = 9
  };
  

public:

  DataRecord();

  DataRecord(const RecordType & data) : theData(data) {}

  virtual ~DataRecord() {}

  const RecordType & data() const { return theData; }
  
  recordName type() const;

  std::string print() {
    std::ostringstream str;
    str << reinterpret_cast<const std::bitset<16>&>(theData); 
    return str.str();
  }
  
protected:
  static const int MaxLBFlag       = 2;
  static const int controlWordFlag = 3;

  static const int BXFlag                  = 1;
  static const int StartOfLBInputDataFlag  = 7;
  static const int EmptyOrDCCDiscardedFlag = 5;
  static const int RMBDiscardedDataFlag    = 6;
  static const int RMBCorruptedDataFlag    = 4;
  static const int RMBDisabledDataFlag     = 161;

  static const int EmptyWordFlag    = 0;
  static const int DCCDiscardedFlag = 1;

  static const int RPC_RECORD_BIT_SIZE = 16;

  static const int RECORD_TYPE_MASK  = 0X3;
  static const int RECORD_TYPE_SHIFT = 14;

  static const int BX_TYPE_MASK  = 0X3;
  static const int BX_TYPE_SHIFT = 12;

  static const int CONTROL_TYPE_MASK  = 0X7;
  static const int CONTROL_TYPE_SHIFT = 11;

  static const int EMPTY_OR_DCCDISCARDED_MASK  = 0X1;
  static const int EMPTY_OR_DCCDISCARDED_SHIFT = 0;

  static const int RMB_DISABLED_MASK  = 0X8;
  static const int RMB_DISABLED_SHIFT = 6;

protected:
  RecordType theData;

};
}
#endif 
