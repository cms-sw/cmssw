#ifndef DQM_RPCMonitorModule_RPCRawDataCountsHistoMaker_H
#define DQM_RPCMonitorModule_RPCRawDataCountsHistoMaker_H

#include <map>


class RPCRawDataCounts;
class TH1F;
class TH2F;



class RPCRawDataCountsHistoMaker {
public:
  RPCRawDataCountsHistoMaker(const RPCRawDataCounts & counts) : theCounts(counts) {}
  
  void fillRecordTypeHisto(int fedId, TH1F* histo) const;
  void fillReadoutErrorHisto(int fedId, TH1F* histo) const;
  void fillGoodEventsHisto(TH2F* histo) const;
  void fillBadEventsHisto(TH2F* histo) const;
  
  static TH1F * emptyRecordTypeHisto(int fedId);
  static TH1F * emptyReadoutErrorHisto(int fedId);

  std::map< std::pair<int,int>, int > readoutErrors(void);
  std::map< std::pair<int,int>, int > recordTypes(void);

private:
  const  RPCRawDataCounts & theCounts;
};

#endif

