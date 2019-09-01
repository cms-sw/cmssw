#ifndef DQM_RPCMonitorModule_RPCRawDataCountsHistoMaker_H
#define DQM_RPCMonitorModule_RPCRawDataCountsHistoMaker_H

#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"

class TH1F;
class TH2F;

class RPCRawDataCountsHistoMaker {
public:
  static TH1F* emptyRecordTypeHisto(int fedId);
  static TH1F* emptyReadoutErrorHisto(int fedId);
  static TH2F* emptyReadoutErrorMapHisto(int fedId, int type);
};

#endif
