#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include "TH2D.h"
#include "TH1D.h"


using namespace std;

struct lessVectSum {
  bool operator () (const pair<unsigned int, unsigned int> & o1,
                    const pair<unsigned int, unsigned int> & o2) {
    return o1.second < o2.second;
  };
};



bool RPCRawSynchro::lessLB::operator () (const LinkBoardElectronicIndex& lb1, 
      const LinkBoardElectronicIndex& lb2) const
{
  if (lb1.dccId < lb2.dccId) return true;
  if (    (lb1.dccId==lb2.dccId)
       && (lb1.dccInputChannelNum< lb2.dccInputChannelNum) ) return true;
  if (    (lb1.dccId==lb2.dccId)
       && (lb1.dccInputChannelNum == lb2.dccInputChannelNum)
       && (lb1.tbLinkInputNum < lb2.tbLinkInputNum ) ) return true;
  if (    (lb1.dccId==lb2.dccId)
       && (lb1.dccInputChannelNum == lb2.dccInputChannelNum)
       && (lb1.tbLinkInputNum == lb2.tbLinkInputNum )
       && (lb1.lbNumInLink < lb2.lbNumInLink) ) return true;
  return false;
}

void RPCRawSynchro::add(const ProdItem & vItem)
{
  for (ProdItem::const_iterator it = vItem.begin(); it != vItem.end(); ++it) { 
    const LinkBoardElectronicIndex & key = it->first;
    int bxDiff = it->second;
    if (theSynchroCounts.find(key)==theSynchroCounts.end()) theSynchroCounts[key]=LinkSynchroCounts();
    theSynchroCounts[key].add(bxDiff);
  }
}

void RPCRawSynchro::LinkSynchroCounts::add(int bxDiff)
{
  if (bxDiff < 0 || bxDiff > 7) return;
  theCounts[bxDiff]++;
}

int RPCRawSynchro::LinkSynchroCounts::mom0() const
{ int result = 0; for (int i=0; i<8; ++i) result += theCounts[i]; return result; }

double RPCRawSynchro::LinkSynchroCounts::mom1() const
{ double result = 0.; for (int i=0; i<8; ++i) result += i*theCounts[i]; return result; }

double RPCRawSynchro::LinkSynchroCounts::mean() const
{ int sum = mom0(); return sum==0 ? 0. : mom1()/sum; }

double RPCRawSynchro::LinkSynchroCounts::rms() const
{
  double result = 0.;
  int      sum = mom0();
  if (sum==0) return 0.;
  double mean = mom1()/sum; 
  for (int i=0; i<8; ++i) result += theCounts[i]*(mean-i)*(mean-i);
  result /= sum;
  return sqrt(result);
}

string RPCRawSynchro::LinkSynchroCounts::print() const
{
  std::ostringstream str;
  for (int i=0; i<8; ++i) str<<" "<<setw(6)<<theCounts[i];
  return str.str();
}


