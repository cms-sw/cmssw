#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroStat.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>

bool RPCLinkSynchroStat::LessLinkName::operator()(const BoardAndCounts& o1, const BoardAndCounts& o2) {
  return o1.first < o2.first;
}
bool RPCLinkSynchroStat::LessCountSum::operator()(const BoardAndCounts& o1, const BoardAndCounts& o2) {
  return o1.second.sum() < o2.second.sum();
}

void RPCLinkSynchroStat::add(const std::string& lbName, const unsigned int* hits) {
  LinkBoard lb(lbName);
  SynchroCounts counts(hits);
  for (auto& it : theLinkStatMap)
    if (it.first == lb)
      it.second += counts;
}

int RPCLinkSynchroStat::LinkBoard::add(const ChamberAndPartition& part) {
  for (const auto& theChamberAndPartition : theChamberAndPartitions) {
    if (theChamberAndPartition == part)
      return 1;
  }
  theChamberAndPartitions.push_back(part);
  return 0;
}

int RPCLinkSynchroStat::LinkBoard::add(const LinkBoardElectronicIndex& ele) {
  for (auto theElePath : theElePaths) {
    if (theElePath.dccId == ele.dccId && theElePath.dccInputChannelNum == ele.dccInputChannelNum &&
        theElePath.tbLinkInputNum == ele.tbLinkInputNum && theElePath.lbNumInLink == ele.lbNumInLink)
      return 1;
  }
  theElePaths.push_back(ele);
  return 0;
}

unsigned int RPCLinkSynchroStat::SynchroCounts::firstHit() const {
  for (unsigned int i = 0; i < 8; ++i)
    if (theCounts[i])
      return i;
  return 8;
}

void RPCLinkSynchroStat::SynchroCounts::set(unsigned int bxDiff) {
  if (bxDiff < 8)
    theCounts[bxDiff] = 1;
}

void RPCLinkSynchroStat::SynchroCounts::increment(unsigned int bxDiff) {
  if (bxDiff < 8)
    theCounts[bxDiff]++;
}

RPCLinkSynchroStat::SynchroCounts& RPCLinkSynchroStat::SynchroCounts::operator+=(const SynchroCounts& rhs) {
  for (unsigned int i = 0; i < 8; ++i)
    theCounts[i] += rhs.theCounts[i];
  return *this;
}

unsigned int RPCLinkSynchroStat::SynchroCounts::mom0() const {
  unsigned int result = 0;
  for (unsigned int i = 0; i < 8; ++i)
    result += theCounts[i];
  return result;
}

double RPCLinkSynchroStat::SynchroCounts::mom1() const {
  double result = 0.;
  for (unsigned int i = 0; i < 8; ++i)
    result += i * theCounts[i];
  return result;
}

double RPCLinkSynchroStat::SynchroCounts::mean() const {
  unsigned int sum = mom0();
  return sum == 0 ? 0. : mom1() / sum;
}

double RPCLinkSynchroStat::SynchroCounts::rms() const {
  double result = 0.;
  int sum = mom0();
  if (sum == 0)
    return 0.;
  double mean = mom1() / sum;
  for (int i = 0; i < 8; ++i)
    result += theCounts[i] * (mean - i) * (mean - i);
  result /= sum;
  return sqrt(result);
}

std::string RPCLinkSynchroStat::SynchroCounts::print() const {
  std::ostringstream str;
  str << " mean: " << std::setw(8) << mean();
  str << " rms: " << std::setw(8) << rms();
  str << " counts:";
  for (int i = 0; i < 8; ++i)
    str << " " << std::setw(4) << theCounts[i];
  return str.str();
}

bool RPCLinkSynchroStat::SynchroCounts::operator==(const SynchroCounts& o) const {
  for (unsigned int idx = 0; idx < 8; ++idx)
    if (theCounts[idx] != o.theCounts[idx])
      return false;
  return true;
}

RPCLinkSynchroStat::RPCLinkSynchroStat(bool useFirstFitOnly) : theUseFirstHitOnly(useFirstFitOnly) {
  for (unsigned int i1 = 0; i1 <= MAXDCCINDEX; ++i1) {
    for (unsigned int i2 = 0; i2 <= MAXRBCINDEX; i2++) {
      for (unsigned int i3 = 0; i3 <= MAXLINKINDEX; ++i3) {
        for (unsigned int i4 = 0; i4 <= MAXLBINDEX; ++i4) {
          theLinkStatNavi[i1][i2][i3][i4] = 0;
        }
      }
    }
  }
  theLinkStatMap.push_back(std::make_pair(LinkBoard("Dummy"), SynchroCounts()));
}

void RPCLinkSynchroStat::init(const RPCReadOutMapping* theCabling, bool addChamberInfo) {
  if (!theCabling)
    return;
  std::vector<const DccSpec*> dccs = theCabling->dccList();
  for (auto dcc : dccs) {
    const std::vector<TriggerBoardSpec>& rmbs = dcc->triggerBoards();
    for (const auto& rmb : rmbs) {
      const std::vector<LinkConnSpec>& links = rmb.linkConns();
      for (const auto& link : links) {
        const std::vector<LinkBoardSpec>& lbs = link.linkBoards();
        for (const auto& lb : lbs) {
          LinkBoardElectronicIndex ele = {
              dcc->id(), rmb.dccInputChannelNum(), link.triggerBoardInputNumber(), lb.linkBoardNumInLink()};
          LinkBoard linkBoard(lb.linkBoardName());
          BoardAndCounts candid = std::make_pair(linkBoard, SynchroCounts());
          std::vector<BoardAndCounts>::iterator candid_place =
              lower_bound(theLinkStatMap.begin(), theLinkStatMap.end(), candid, LessLinkName());
          if (candid_place != theLinkStatMap.end() && candid.first == candid_place->first) {
            candid_place->first.add(ele);
          } else {
            candid_place = theLinkStatMap.insert(candid_place, candid);
            candid_place->first.add(ele);
            if (addChamberInfo) {
              const std::vector<FebConnectorSpec>& febs = lb.febs();
              for (const auto& feb : febs) {
                std::string chamberName = feb.chamber().chamberLocationName();
                std::string partitionName = feb.feb().localEtaPartitionName();
                LinkBoard::ChamberAndPartition chamberAndPartition = std::make_pair(chamberName, partitionName);
                candid_place->first.add(chamberAndPartition);
              }
            }
          }
        }
      }
    }
  }
  for (unsigned int idx = 0; idx < theLinkStatMap.size(); ++idx) {
    const std::vector<LinkBoardElectronicIndex>& paths = theLinkStatMap[idx].first.paths();
    for (auto path : paths) {
      theLinkStatNavi[path.dccId - DCCINDEXSHIFT][path.dccInputChannelNum][path.tbLinkInputNum][path.lbNumInLink] = idx;
    }
  }
  //  LogTrace("RPCLinkSynchroStat") <<" SIZE OF LINKS IS: " << theLinkStatMap.size() << endl;
}

void RPCLinkSynchroStat::add(const RPCRawSynchro::ProdItem& vItem, std::vector<LinkBoardElectronicIndex>& problems) {
  std::vector<int> hits(theLinkStatMap.size(), 0);
  std::vector<ShortLinkInfo> slis;
  for (const auto& it : vItem) {
    const LinkBoardElectronicIndex& path = it.first;
    unsigned int bxDiff = it.second;
    unsigned int eleCode = (path.dccId - DCCINDEXSHIFT) * 100000 + path.dccInputChannelNum * 1000 +
                           path.tbLinkInputNum * 10 + path.lbNumInLink;
    unsigned int idx =
        theLinkStatNavi[path.dccId - DCCINDEXSHIFT][path.dccInputChannelNum][path.tbLinkInputNum][path.lbNumInLink];
    if (hits[idx] == 0) {
      ShortLinkInfo sli = {idx, std::vector<unsigned int>(1, eleCode), SynchroCounts()};
      slis.push_back(sli);
      hits[idx] = slis.size();
    } else {
      std::vector<unsigned int>& v = slis[hits[idx] - 1].hit_paths;
      std::vector<unsigned int>::iterator iv = lower_bound(v.begin(), v.end(), eleCode);
      if (iv == v.end() || (*iv) != eleCode)
        v.insert(iv, eleCode);
    }
    slis[hits[idx] - 1].counts.set(bxDiff);  // ensure one count per LB per BX
  }

  for (const auto& sli : slis) {
    if (theUseFirstHitOnly) {
      theLinkStatMap[sli.idx].second.increment(sli.counts.firstHit());  // first hit only
    } else {
      theLinkStatMap[sli.idx].second += sli.counts;
    }
    if (theLinkStatMap[sli.idx].first.paths().size() != sli.hit_paths.size()) {
      const std::vector<LinkBoardElectronicIndex>& paths = theLinkStatMap[sli.idx].first.paths();
      problems.insert(problems.end(), paths.begin(), paths.end());
    }
  }
}

std::string RPCLinkSynchroStat::dumpDelays() {
  std::ostringstream str;
  std::vector<BoardAndCounts> sortedStat = theLinkStatMap;
  stable_sort(sortedStat.begin(), sortedStat.end(), LessCountSum());
  for (auto& idx : sortedStat) {
    const LinkBoard& board = idx.first;
    const SynchroCounts& counts = idx.second;

    // DUMP LINKNAME
    str << std::setw(20) << board.name();

    // DUMP COUNTS
    str << " " << counts.print();

    //PATHS
    str << " paths: ";
    const std::vector<LinkBoardElectronicIndex>& paths = board.paths();
    for (auto path : paths)
      str << "{" << path.dccId << "," << std::setw(2) << path.dccInputChannelNum << "," << std::setw(2)
          << path.tbLinkInputNum << "," << path.lbNumInLink << "}";

    // DUMP CHAMBERS
    std::map<std::string, std::vector<std::string> > chMap;
    const std::vector<LinkBoard::ChamberAndPartition>& chamberAndPartitions = board.chamberAndPartitions();
    for (const auto& chamberAndPartition : chamberAndPartitions) {
      std::vector<std::string>& partitions = chMap[chamberAndPartition.first];
      if (find(partitions.begin(), partitions.end(), chamberAndPartition.second) == partitions.end())
        partitions.push_back(chamberAndPartition.second);
    }
    str << " chambers: ";
    for (std::map<std::string, std::vector<std::string> >::const_iterator im = chMap.begin(); im != chMap.end(); ++im) {
      str << im->first << "(";
      for (std::vector<std::string>::const_iterator ip = im->second.begin(); ip != im->second.end(); ++ip) {
        str << *ip;
        if ((ip + 1) != (im->second.end()))
          str << ",";
        else
          str << ")";
      }
    }

    str << std::endl;
  }
  LogTrace("RPCLinkSynchroStat") << "RPCLinkSynchroStat::dumpDelays,  SIZE OF LINKS IS: " << theLinkStatMap.size()
                                 << std::endl;
  return str.str();
}
