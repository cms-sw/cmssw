#ifndef DataFormats_RPCDigi_RPCRawDataCounts_H
#define DataFormats_RPCDigi_RPCRawDataCounts_H

#include <map>
#include <vector>
#include <string>

namespace rpcrawtodigi { class DataRecord; }
namespace rpcrawtodigi { class ReadoutError; }

class RPCRawDataCounts {
public:

  RPCRawDataCounts() {}
  ~RPCRawDataCounts() { }
  void addDccRecord(int fedId, const rpcrawtodigi::DataRecord & record, int weight=1);
  void addReadoutError(int fedId, const rpcrawtodigi::ReadoutError & error, int weight=1);
  void operator+= (const RPCRawDataCounts& );
  std::string print() const;

  int fedBxRecords(int fedId) const; 
  int fedFormatErrors(int fedId) const;
  int fedErrorRecords(int fedId) const;

private:

  friend class  RPCMonitorRaw;
  std::map< std::pair<int,int>, int> theRecordTypes;
  std::map< std::pair<int,int>, int> theReadoutErrors; 
  std::map< std::pair<int,int>, int> theGoodEvents;
  std::map< std::pair<int,int>, int> theBadEvents;

};
#endif
