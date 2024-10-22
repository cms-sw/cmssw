#ifndef DataFormats_RPCDigi_DataRecord_H
#define DataFormats_RPCDigi_DataRecord_H

#include <string>
#include <bitset>
#include <sstream>
#include <cstdint>

namespace rpcrawtodigi {
  class DataRecord {
  public:
    typedef uint16_t Data;
    enum DataRecordType {
      None = 0,
      StartOfBXData = 1,
      StartOfTbLinkInputNumberData = 2,
      ChamberData = 3,
      Empty = 4,
      RDDM = 5,
      SDDM = 6,
      RCDM = 7,
      RDM = 8,
      UndefinedType = 9
    };

  public:
    explicit DataRecord(const Data& data = None) : theData(data) {}

    virtual ~DataRecord() {}

    const Data& data() const { return theData; }

    DataRecordType type() const;

    static std::string name(const DataRecordType& code);

    std::string print() const {
      std::ostringstream str;
      str << std::bitset<16>(theData);
      return str.str();
    }

    static std::string print(const DataRecord& record);

  protected:
    Data theData;
  };
}  // namespace rpcrawtodigi
#endif
