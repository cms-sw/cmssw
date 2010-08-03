#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroStat.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace std;

bool RPCLinkSynchroStat::LessLinkName::operator()(const BoardAndCounts& o1, const BoardAndCounts& o2) { return o1.first < o2.first; }
bool RPCLinkSynchroStat::LessCountSum::operator()(const BoardAndCounts& o1, const BoardAndCounts& o2) { return o1.second.sum()< o2.second.sum(); }

int RPCLinkSynchroStat::LinkBoard::add(const ChamberAndPartition & part) 
{
  for (std::vector<ChamberAndPartition>::const_iterator it=theChamberAndPartitions.begin(); it != theChamberAndPartitions.end(); ++it) {
    if( (*it)==part) return 1;
  }
  theChamberAndPartitions.push_back(part);
  return 0;
}

int RPCLinkSynchroStat::LinkBoard::add(const LinkBoardElectronicIndex & ele) 
{
  for (std::vector<LinkBoardElectronicIndex>::const_iterator it=theElePaths.begin(); it != theElePaths.end(); ++it) {
    if (    it->dccId == ele.dccId
         && it->dccInputChannelNum == ele.dccInputChannelNum
         && it->tbLinkInputNum == ele.tbLinkInputNum
         && it->lbNumInLink == ele.lbNumInLink) return 1; 
  }
  theElePaths.push_back(ele);
  return 0;
}

unsigned int RPCLinkSynchroStat::SynchroCounts::firstHit() const
{
  for (unsigned int i=0; i <8 ; ++i) if(theCounts[i]) return i;
  return 8;
}

void  RPCLinkSynchroStat::SynchroCounts::set(unsigned int bxDiff)
{
  if (bxDiff < 8) theCounts[bxDiff]=1;
}

void  RPCLinkSynchroStat::SynchroCounts::increment(unsigned int bxDiff)
{
  if (bxDiff < 8) theCounts[bxDiff]++;
}

RPCLinkSynchroStat::SynchroCounts & RPCLinkSynchroStat::SynchroCounts::operator+=(const SynchroCounts &rhs)
{
  for (unsigned int i=0; i<8; ++i) theCounts[i]+=rhs.theCounts[i];
  return *this;
}

unsigned int  RPCLinkSynchroStat::SynchroCounts::mom0() const
{ unsigned int result = 0; for (unsigned int i=0; i<8; ++i) result += theCounts[i]; return result; }

double  RPCLinkSynchroStat::SynchroCounts::mom1() const
{ double result = 0.; for (unsigned int i=0; i<8; ++i) result += i*theCounts[i]; return result; }

double  RPCLinkSynchroStat::SynchroCounts::mean() const
{ unsigned int sum = mom0(); return sum==0 ? 0. : mom1()/sum; }

double  RPCLinkSynchroStat::SynchroCounts::rms() const
{
  double result = 0.;
  int      sum = mom0();
  if (sum==0) return 0.;
  double mean = mom1()/sum;
  for (int i=0; i<8; ++i) result += theCounts[i]*(mean-i)*(mean-i);
  result /= sum;
  return sqrt(result);
}

string  RPCLinkSynchroStat::SynchroCounts::print() const
{
  std::ostringstream str;
  str<<" mean: "<<setw(8)<<mean();
  str<<" rms: "<<setw(8)<<rms();
  str<<" counts:"; for (int i=0; i<8; ++i) str<<" "<<setw(4)<<theCounts[i];
  return str.str();
}

bool RPCLinkSynchroStat::SynchroCounts::operator==(const SynchroCounts &o) const
{
  for (unsigned int idx=0; idx <8; ++idx) if (theCounts[idx] != o.theCounts[idx]) return false;
  return true;
}





RPCLinkSynchroStat::RPCLinkSynchroStat(bool useFirstFitOnly)
  : theUseFirstHitOnly(useFirstFitOnly)
{
  for (unsigned int i1=0; i1<=MAXDCCINDEX; ++i1) {
    for (unsigned int i2=0; i2<=MAXRBCINDEX; i2++) {
      for (unsigned int i3=0; i3 <=MAXLINKINDEX; ++i3) {
        for (unsigned int i4=0; i4<=MAXLBINDEX; ++i4) {
          theLinkStatNavi[i1][i2][i3][i4]=0;
        }
      }
    }
  }
  theLinkStatMap.push_back( make_pair(LinkBoard("Dummy"), SynchroCounts()) );
}

void RPCLinkSynchroStat::init(const RPCReadOutMapping* theCabling, bool addChamberInfo)
{
  if (!theCabling) return;
  vector<const DccSpec*> dccs = theCabling->dccList();
  for (vector<const DccSpec*>::const_iterator it1= dccs.begin(); it1!= dccs.end(); ++it1) {
    const vector<TriggerBoardSpec> & rmbs = (*it1)->triggerBoards();
    for (vector<TriggerBoardSpec>::const_iterator it2 = rmbs.begin(); it2 != rmbs.end(); ++it2) {
      const  vector<LinkConnSpec> & links = it2->linkConns();
      for (vector<LinkConnSpec>::const_iterator it3 = links.begin(); it3 != links.end(); ++it3) {
        const  vector<LinkBoardSpec> & lbs = it3->linkBoards();
        for (vector<LinkBoardSpec>::const_iterator it4=lbs.begin(); it4 != lbs.end(); ++it4) {
          LinkBoardElectronicIndex ele = { (*it1)->id(), it2->dccInputChannelNum(), it3->triggerBoardInputNumber(), it4->linkBoardNumInLink()}; 
          LinkBoard linkBoard(it4->linkBoardName());
          BoardAndCounts candid = make_pair(linkBoard,SynchroCounts());              
          vector<BoardAndCounts>::iterator candid_place = lower_bound(theLinkStatMap.begin(), theLinkStatMap.end(), candid, LessLinkName());
          if (candid_place != theLinkStatMap.end() && candid.first == candid_place->first) {
            candid_place->first.add(ele);
          } 
          else {
            candid_place = theLinkStatMap.insert(candid_place,candid);
            candid_place->first.add(ele);
            if (addChamberInfo) {
            const  vector<FebConnectorSpec> & febs = it4->febs(); 
            for (vector<FebConnectorSpec>::const_iterator it5=febs.begin(); it5!= febs.end(); ++it5) {
              string chamberName = it5->chamber().chamberLocationName();
              string partitionName = it5->feb().localEtaPartitionName();
              LinkBoard::ChamberAndPartition chamberAndPartition = make_pair(chamberName, partitionName);
              candid_place->first.add(chamberAndPartition);
            }
            } 
          }
        }
      } 
    }
  }
  for (unsigned int idx=0; idx<theLinkStatMap.size(); ++idx) {
    const vector<LinkBoardElectronicIndex> &paths= theLinkStatMap[idx].first.paths();
    for  (vector<LinkBoardElectronicIndex>::const_iterator it=paths.begin(); it!= paths.end();++it) {
      theLinkStatNavi[it->dccId-DCCINDEXSHIFT][it->dccInputChannelNum][it->tbLinkInputNum][it->lbNumInLink]=idx;
    }
  }
//  LogTrace("RPCLinkSynchroStat") <<" SIZE OF LINKS IS: " << theLinkStatMap.size() << endl;
}

void RPCLinkSynchroStat::add(const RPCRawSynchro::ProdItem & vItem, vector<LinkBoardElectronicIndex> & problems)
{
  vector< int > hits(theLinkStatMap.size(),0);
  vector<ShortLinkInfo> slis;
  for ( RPCRawSynchro::ProdItem::const_iterator it = vItem.begin(); it != vItem.end(); ++it) {
    const LinkBoardElectronicIndex & path = it->first;
    unsigned int bxDiff = it->second;
    unsigned int eleCode = (path.dccId-DCCINDEXSHIFT)*100000 + path.dccInputChannelNum*1000 + path.tbLinkInputNum*10+path.lbNumInLink;
    unsigned int idx = theLinkStatNavi[path.dccId-DCCINDEXSHIFT][path.dccInputChannelNum][path.tbLinkInputNum][path.lbNumInLink];
    if  (hits[idx]==0) {
      ShortLinkInfo sli = { idx, vector<unsigned int>(1, eleCode), SynchroCounts() };
      slis.push_back( sli );
      hits[idx]=slis.size();
    } 
    else {
      vector<unsigned int> & v = slis[hits[idx]-1].hit_paths;
      vector<unsigned int>::iterator iv = lower_bound (v.begin(), v.end(), eleCode);
      if (iv == v.end() || (*iv) != eleCode) v.insert(iv,eleCode);
    } 
    slis[hits[idx]-1].counts.set(bxDiff); // ensure one count per LB per BX
  }

  for (vector<ShortLinkInfo>::const_iterator ic = slis.begin(); ic !=slis.end(); ++ic) {
    if (theUseFirstHitOnly) {
      theLinkStatMap[ic->idx].second.increment( ic->counts.firstHit() );  // first hit only  
    } else  {
      theLinkStatMap[ic->idx].second += ic->counts;
    }
    if (theLinkStatMap[ ic->idx].first.paths().size() != ic->hit_paths.size()) {
       const vector<LinkBoardElectronicIndex> & paths =  theLinkStatMap[ ic->idx].first.paths();
       problems.insert(problems.end(),paths.begin(),paths.end());
    }
  }

}

string RPCLinkSynchroStat::dumpDelays() 
{
  std::ostringstream str;
  std::vector<BoardAndCounts> sortedStat = theLinkStatMap;
  stable_sort(sortedStat.begin(),sortedStat.end(),LessCountSum()); 
  for (unsigned int idx=0; idx<sortedStat.size(); ++idx) {
    const LinkBoard & board = sortedStat[idx].first;
    const SynchroCounts & counts = sortedStat[idx].second;

    // DUMP LINKNAME
    str << setw(20) << board.name();

    // DUMP COUNTS
    str <<" "<<counts.print();

    //PATHS
    str <<" paths: ";
    const std::vector<LinkBoardElectronicIndex> & paths=board.paths();
    for (std::vector<LinkBoardElectronicIndex>::const_iterator ip=paths.begin(); ip!=paths.end();++ip) 
        str<<"{"<<ip->dccId<<","<<setw(2)<<ip->dccInputChannelNum<<","<<setw(2)<<ip->tbLinkInputNum<<","<<ip->lbNumInLink<<"}";

    // DUMP CHAMBERS
    map<string,vector<string> > chMap;
    const std::vector<LinkBoard::ChamberAndPartition> & chamberAndPartitions = board.chamberAndPartitions();
    for (vector<LinkBoard::ChamberAndPartition>::const_iterator it=chamberAndPartitions.begin(); it!=chamberAndPartitions.end();++it) {
      vector<string> & partitions = chMap[it->first];
      if (find(partitions.begin(), partitions.end(), it->second) == partitions.end()) partitions.push_back(it->second);
    }
    str << " chambers: ";
    for (map<string,vector<string> >::const_iterator im=chMap.begin(); im != chMap.end(); ++im) {
      str <<im->first<<"(";
      for (vector<string>::const_iterator ip=im->second.begin(); ip != im->second.end(); ++ip) { str << *ip; if ((ip+1)!= (im->second.end()) ) str<<","; else str <<")"; }
    }

    
    str <<endl;
  }
  LogTrace("RPCLinkSynchroStat") <<"RPCLinkSynchroStat::dumpDelays,  SIZE OF LINKS IS: " << theLinkStatMap.size() << endl;
  return str.str();
}
