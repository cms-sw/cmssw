#ifndef DataFormats_RPCDigi_EmptyWord_H
#define DataFormats_RPCDigi_EmptyWord_H

#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"
#include <string>

namespace rpcrawtodigi{
class EmptyWord : public DataRecord {
private:
  static const int  EW_TYPE = 0xE800;
public:
  EmptyWord() : DataRecord(EW_TYPE) {}
  virtual ~EmptyWord(){}
  std::string print()  const { return " EMPTY "; }
  static bool matchType(const DataRecord & record) { return record.data()==EW_TYPE; }
};
}
#endif 
