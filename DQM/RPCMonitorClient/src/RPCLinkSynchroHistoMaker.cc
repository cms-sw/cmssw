#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroHistoMaker.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include <sstream>
#include "TH1F.h"
#include "TH2F.h"
#include <algorithm>
#include <iostream>

struct OrderLbSpread {
  bool operator()(const std::pair<double, unsigned int>& lb1, const std::pair<double, unsigned int>& lb2) {
    return lb1.first > lb2.first;
  }
};
struct OrderLbOccup {
  bool operator()(const std::pair<unsigned int, unsigned int>& lb1, const std::pair<unsigned int, unsigned int>& lb2) {
    return lb1.first > lb2.first;
  }
};

void RPCLinkSynchroHistoMaker::fill(TH1F* hDelay, TH2F* hDelaySpread, TH2F* hTopOccup, TH2F* hTopSpread) const {
  hDelay->Reset();
  hDelaySpread->Reset();
  hTopOccup->Reset();
  hTopSpread->Reset();

  typedef std::vector<std::pair<unsigned int, unsigned int> > TopOccup;
  typedef std::vector<std::pair<double, unsigned int> > TopSpread;
  TopOccup topOccup(10, std::make_pair(0, 0));
  TopSpread topSpread(10, std::make_pair(0., 0));

  for (unsigned int idx = 0; idx < theLinkStat.theLinkStatMap.size(); ++idx) {
    const RPCLinkSynchroStat::BoardAndCounts& bc = theLinkStat.theLinkStatMap[idx];

    int sum = bc.second.sum();
    double rms = bc.second.rms();

    hDelaySpread->Fill(bc.second.mean() - 3., bc.second.rms());

    if (sum == 0)
      continue;
    for (int i = 0; i <= 7; ++i)
      hDelay->Fill(i - 3, bc.second.counts()[i]);

    std::pair<unsigned int, unsigned int> canOccup = std::make_pair(sum, idx);
    std::pair<double, unsigned int> canSpread = std::make_pair(rms, idx);
    TopOccup::iterator io = upper_bound(topOccup.begin(), topOccup.end(), canOccup, OrderLbOccup());
    TopSpread::iterator is = upper_bound(topSpread.begin(), topSpread.end(), canSpread, OrderLbSpread());
    if (io != topOccup.end()) {
      topOccup.insert(io, canOccup);
      topOccup.erase(topOccup.end() - 1);
    }
    if (is != topSpread.end()) {
      topSpread.insert(is, canSpread);
      topSpread.erase(topSpread.end() - 1);
    }
  }

  for (int itop = 0; itop < 10; itop++) {
    const RPCLinkSynchroStat::BoardAndCounts& occup = theLinkStat.theLinkStatMap[topOccup[itop].second];
    const RPCLinkSynchroStat::BoardAndCounts& spread = theLinkStat.theLinkStatMap[topSpread[itop].second];
    hTopOccup->GetYaxis()->SetBinLabel(itop + 1, occup.first.name().c_str());
    hTopSpread->GetYaxis()->SetBinLabel(itop + 1, spread.first.name().c_str());
    for (unsigned int icount = 0; icount < occup.second.counts().size(); icount++) {
      hTopOccup->SetBinContent(icount + 1, itop + 1, float(occup.second.counts()[icount]));
      hTopSpread->SetBinContent(icount + 1, itop + 1, float(spread.second.counts()[icount]));
    }
  }
  //  for (int j=0; j<10; j++) { cout <<"topSpread["<<j<<"] = "<<topSpread[j].first<<endl; }
  //  for (int j=0; j<10; j++) { cout <<"topOccup["<<j<<"] = "<<topOccup[j].first<<endl; }
}
