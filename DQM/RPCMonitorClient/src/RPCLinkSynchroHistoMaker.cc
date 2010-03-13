#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroHistoMaker.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include <sstream>
#include "TH1F.h"
#include "TH2F.h"
#include <algorithm>
#include <iostream>

using namespace std;

void RPCLinkSynchroHistoMaker::fillDelayHisto(TH1F* histo)
{
  histo->Reset();
  for (std::vector<RPCLinkSynchroStat::BoardAndCounts>::const_iterator it = theLinkStat.theLinkStatMap.begin(); it != theLinkStat.theLinkStatMap.end(); ++it) {
    for (int i=0; i<=7; ++i) histo->Fill(i-3,it->second.counts()[i]);
  } 
}

void RPCLinkSynchroHistoMaker::fillDelaySpreadHisto(TH2F* histo)
{
  histo->Reset();
  for (std::vector<RPCLinkSynchroStat::BoardAndCounts>::const_iterator it = theLinkStat.theLinkStatMap.begin(); it != theLinkStat.theLinkStatMap.end(); ++it) {
    histo->Fill(it->second.mean()-3.,it->second.rms());
  } 
}

