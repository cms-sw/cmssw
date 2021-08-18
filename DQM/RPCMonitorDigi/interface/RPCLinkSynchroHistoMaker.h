#ifndef DQM_RPCMonitorDigi_RPCLinkSynchroHistoMaker_H
#define DQM_RPCMonitorDigi_RPCLinkSynchroHistoMaker_H

#include "DQM/RPCMonitorDigi/interface/RPCLinkSynchroStat.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include <TH1F.h>
#include <TH2F.h>

#include <string>

class RPCLinkSynchroHistoMaker {
public:
  RPCLinkSynchroHistoMaker(const RPCLinkSynchroStat& a) : theLinkStat(a) {}
  void fillDelaySpreadHisto(TH2F* histo);
  void fillDelayHisto(TH1F* histo);
  void fill(TH1F* hDelay, TH2F* hDelaySpread, TH2F* hTopOccup, TH2F* hTopSpread) const;

private:
  const RPCLinkSynchroStat& theLinkStat;
};
#endif
