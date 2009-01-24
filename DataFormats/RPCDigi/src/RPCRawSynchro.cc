#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"

#include <iostream>
#include <fstream>
#include <iomanip>
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
    if (theSynchroCounts.find(key)==theSynchroCounts.end()) theSynchroCounts[key]=vector<int>(8,0);
    vector<int> & v = theSynchroCounts[key];
    if (bxDiff < 0) continue;
    if (bxDiff > 7) continue;
    v[bxDiff]++;
  }
}
