#ifndef EventFilter_RPCRawToDigi_RPCRawDataCounts_H
#define EventFilter_RPCRawToDigi_RPCRawDataCounts_H

#include <map>
#include <vector>
#include <string>

namespace rpcrawtodigi { class DataRecord; }
namespace rpcrawtodigi { class ReadoutError; }
class TH1F;
class TH2F;

class RPCRawDataCounts {
public:

  RPCRawDataCounts() {}
  ~RPCRawDataCounts() { }
  void addDccRecord(int fedId, const rpcrawtodigi::DataRecord & record, int weight=1);
  void addReadoutError(int fedId, const rpcrawtodigi::ReadoutError & error, int weight=1);
  void operator+= (const RPCRawDataCounts& );
  std::string print() const;

  void fillRecordTypeHisto(int fedId, TH1F* histo) const;
  void fillReadoutErrorHisto(int fedId, TH1F* histo) const;
  void fillGoodEventsHisto(TH2F* histo) const;
  void fillBadEventsHisto(TH2F* histo) const;
  
  TH1F * emptyRecordTypeHisto(int fedId) const;
  TH1F * emptyReadoutErrorHisto(int fedId) const;

private:

  std::map< std::pair<int,int>, int> theRecordTypes;
  std::map< std::pair<int,int>, int> theReadoutErrors; 
  std::map< std::pair<int,int>, int> theGoodEvents;
  std::map< std::pair<int,int>, int> theBadEvents;
};
#endif
