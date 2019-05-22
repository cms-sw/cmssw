#ifndef DataFormats_RPCDigi_RecordCD_H
#define DataFormats_RPCDigi_RecordCD_H

#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include <vector>

namespace rpcrawtodigi {
  class RecordCD : public DataRecord {
  private:
    static const int CD_TYPE_LESSTHENFLAG = 0x3;
    static const int CD_TYPE_SHIFT = 14;

    static const int PARTITION_DATA_MASK = 0XFF;
    static const int PARTITION_DATA_SHIFT = 0;

    static const int PARTITION_NUMBER_MASK = 0XF;
    static const int PARTITION_NUMBER_SHIFT = 10;

    static const int HALFP_MASK = 0X1;
    static const int HALFP_SHIFT = 8;

    static const int EOD_MASK = 0X1;
    static const int EOD_SHIFT = 9;

    static const int CHAMBER_MASK = 0X3;
    static const int CHAMBER_SHIFT = 14;

    static const int BITS_PER_PARTITION = 8;

  public:
    // empty record
    RecordCD() : DataRecord() {}

    // set with Data
    RecordCD(int chamber, int partitionNumber, int eod, int halfP, const std::vector<int>& packedStrips);

    // set LB from raw
    RecordCD(const Data& lbData) : DataRecord(lbData) {}

    // specialize given recort to this type
    RecordCD(const DataRecord& rec) : DataRecord(rec) {}

    ~RecordCD() override {}

    static bool matchType(const DataRecord& record) {
      return ((record.data() >> CD_TYPE_SHIFT) < CD_TYPE_LESSTHENFLAG);
    }

    // more precisly - link board in link number
    int lbInLink() const;

    int partitionNumber() const;
    int eod() const;
    int halfP() const;

    std::vector<int> packedStrips() const;
    int partitionData() const;

    std::string print() const;
  };
}  // namespace rpcrawtodigi
#endif
