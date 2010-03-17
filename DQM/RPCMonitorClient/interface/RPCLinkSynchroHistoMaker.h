#ifndef DQM_RPCMonitorClient_RPCLinkSynchroHistoMaker_H
#define DQM_RPCMonitorClient_RPCLinkSynchroHistoMaker_H

#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include <string>
#include <vector>


class TH1F;
class TH2F;
class RPCReadOutMapping;

class RPCLinkSynchroHistoMaker {
public:
  RPCLinkSynchroHistoMaker(const RPCRawSynchro&, const RPCReadOutMapping* rm=0);
  std::string dumpDelays();
  void fillDelaySpreadHisto(TH2F* histo);
//  void fillLinksBadSynchro(TH2F* histo);
//  void fillLinksLowStat(TH2F* histo);
//  void fillLinksMostNoisy(TH2F* histo);
    
private:
  struct LinkStat { std::string nameLink, nameChamber, namePart; 
                    double mean, sum, rms; std::vector<int> vectStat; 
                    std::string print() const; };
  struct LessVectStatSum{ bool operator()(const LinkStat &o1, const LinkStat& o2); };
  struct LessVectStatMean{ bool operator()(const LinkStat &o1, const LinkStat& o2); };
  void makeLinkStats();
private:
  const RPCRawSynchro & theRawSynchro;
  const RPCReadOutMapping * theCabling;
  bool theUpdated;
  std::vector<LinkStat> theLinkStat;
}; 
#endif
