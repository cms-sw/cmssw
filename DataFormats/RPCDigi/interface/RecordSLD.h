#ifndef DataFormats_RPCDigi_RecordSLD_H
#define DataFormats_RPCDigi_RecordSLD_H

#include "DataFormats/RPCDigi/interface/DataRecord.h"

namespace rpcrawtodigi {
  class RecordSLD : public DataRecord {
  private:
    static const int SLD_TYPE_FLAG = 0x1F;
    static const int SLD_TYPE_SHIFT = 11;
    static const int TB_LINK_INPUT_NUMBER_MASK = 0x1F;
    static const int TB_LINK_INPUT_NUMBER_SHIFT = 0;
    static const int TB_RMB_MASK = 0X3F;
    static const int TB_RMB_SHIFT = 5;

  public:
    // empty record
    RecordSLD() : DataRecord() {}

    // set TB
    RecordSLD(int tbLinkInputNumber, int rmb) : DataRecord(0) {
      theData = SLD_TYPE_FLAG << SLD_TYPE_SHIFT;
      theData |= (tbLinkInputNumber << TB_LINK_INPUT_NUMBER_SHIFT);
      theData |= (rmb << TB_RMB_SHIFT);
    }

    // specialize given recort to this type
    RecordSLD(const DataRecord& rec) : DataRecord(rec) {}

    ~RecordSLD() override {}

    int tbLinkInputNumber() const { return (theData >> TB_LINK_INPUT_NUMBER_SHIFT) & TB_LINK_INPUT_NUMBER_MASK; }

    int rmb() const { return (theData >> TB_RMB_SHIFT) & TB_RMB_MASK; }

    static bool matchType(const DataRecord& record) { return (SLD_TYPE_FLAG == (record.data() >> SLD_TYPE_SHIFT)); }
    std::string print() const;
  };
}  // namespace rpcrawtodigi
#endif
