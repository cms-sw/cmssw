#ifndef DQM_RPCMonitorClient_RPCLinkSynchroHistoMaker_H
#define DQM_RPCMonitorClient_RPCLinkSynchroHistoMaker_H

#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroStat.h"
#include <string>
#include <vector>
#include <map>


class TH1F;
class TH2F;
class RPCReadOutMapping;

class RPCLinkSynchroHistoMaker {
public:
  RPCLinkSynchroHistoMaker(const RPCLinkSynchroStat & a) : theLinkStat(a) {}
  void fillDelaySpreadHisto(TH2F* histo);
  void fillDelayHisto(TH1F* histo);
  void fill(TH1F* hDelay, TH2F* hDelaySpread, TH2F* hTopOccup, TH2F* hTopSpread) const;
    
private:
  const RPCLinkSynchroStat & theLinkStat;
}; 
#endif
