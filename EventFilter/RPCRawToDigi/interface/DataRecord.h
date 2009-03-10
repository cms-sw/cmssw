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
    None = 0,
    StartOfBXData = 1,
    StartOfTbLinkInputNumberData = 2,
    ChamberData = 3,
    Empty = 4,
    RDDM = 5,
    SDDM = 6,
    RCDM = 7,
    RDM  = 8, 
    UndefinedType = 9
  };
  

public:

  DataRecord(const RecordType & data = 0) : theData(data) {}

  virtual ~DataRecord() {}

  const RecordType & data() const { return theData; }
  
  recordName type() const;

  static std::string name(const recordName & code);

  std::string print() const {
    std::ostringstream str;
    str << reinterpret_cast<const std::bitset<16>&>(theData); 
    return str.str();
  }

  static std::string print(const DataRecord & data);
  
protected:
  RecordType theData;

};
}
#endif 
