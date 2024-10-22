#ifndef DQM_RPCMonitorModule_RPCRawDataCountsHistoMaker_H
#define DQM_RPCMonitorModule_RPCRawDataCountsHistoMaker_H

#include "TH1F.h"
#include "TH2F.h"

class RPCRawDataCountsHistoMaker {
public:
  static TH1F* emptyRecordTypeHisto(int fedId);
  static TH1F* emptyReadoutErrorHisto(int fedId);
  static TH2F* emptyReadoutErrorMapHisto(int fedId, int type);
};

#endif
