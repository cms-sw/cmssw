#include "EventFilter/RPCRawToDigi/interface/RPCRawSynchro.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "TH2D.h"
#include "TH1D.h"



using namespace rpcrawtodigi;
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

int RPCRawSynchro::bxDifference(const EventRecords & event)
{
  static const int nOrbits = 3564;
  int diff = event.recordBX().bx() - event.triggerBx() + 3;
  if (diff >  nOrbits/2) diff -=nOrbits;
  if (diff < -nOrbits/2) diff +=nOrbits;
  return diff;
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

std::string RPCRawSynchro::dumpDccBx(int dcc, TH1D * histo) const
{
  typedef std::map<int,std::vector<int> > CountMap;
  CountMap bxCounts;
  std::ostringstream str;

  for (int rmb=0; rmb <=35; rmb++)  bxCounts[rmb]=vector<int>(8,0);
  
  for (LBCountMap::const_iterator im = theSynchroCounts.begin(); im != theSynchroCounts.end(); ++im) {
    const LinkBoardElectronicIndex & ele = im->first;
    if (ele.dccId!= dcc) continue;
    vector<int> & v = bxCounts[ele.dccInputChannelNum];
    const vector<int> & vSyncro = im->second;
    for (int i=0; i<8;++i) v[i] += vSyncro[i]; 
  }

  for (CountMap::const_iterator im=bxCounts.begin(); im!= bxCounts.end(); ++im) {
    int totCount=0;
    for (unsigned int i=0; i<im->second.size(); i++) totCount +=  im->second[i];
    if (totCount>0) {
      str<<"dcc="<<setw(3)<<dcc<<" rmb="<<setw(3)<<im->first<<" counts: ";
      for (unsigned int i=0; i<im->second.size(); i++) str<<" "<<setw(6)<<im->second[i];
      for (unsigned int i=0; i<im->second.size(); i++) histo->Fill(double(i)-3., double(im->second[i]));
      str<<endl;
    }
  }
  return str.str();
}

std::string RPCRawSynchro::dumpDelays(const RPCReadOutMapping *rm, TH2D * histo) const
{
  std::ostringstream sss;

  vector< std::vector<int> > vectRes;
  vector< string > vectNamesCh;
  vector< string > vectNamesPart;
  vector< string > vectNamesLink;
  vector< float > vectMean;
  vector< pair<unsigned int, unsigned int> > vectSums;
  vector< float > vectRMS;
  for (LBCountMap::const_iterator im = theSynchroCounts.begin(); im != theSynchroCounts.end(); ++im) {
    const LinkBoardSpec* linkBoard = rm->location(im->first);
    if (linkBoard==0) continue;
    float sumW =0.;
    unsigned  int stat = 0;
    for (unsigned int i=0; i<im->second.size(); i++) {
      stat += im->second[i];
      sumW += i*im->second[i];
    }
    float mean = sumW/stat;
    float rms2 = 0.;
    for (unsigned int i=0; i<im->second.size(); i++) rms2 += im->second[i]*(mean-i)*(mean-i);
//    vector<string> chnames;
//    for (vector<FebConnectorSpec>::const_iterator ifs = linkBoard->febs().begin(); ifs !=  linkBoard->febs().end(); ++ifs) {
//      vector<string>::iterator immm = find(chnames.begin(),chnames.end(),ifs->chamber().chamberLocationName);
//      if (immm == chnames.end()) chnames.push_back(ifs->chamber().chamberLocationName);
//    }
    const ChamberLocationSpec & chamber = linkBoard->febs().front().chamber();
    const FebLocationSpec & feb =  linkBoard->febs().front().feb();
    sss <<chamber.chamberLocationName
        <<" "<< feb.localEtaPartition
        <<" mean: "<<mean <<" rms: " << sqrt(rms2/stat)
        << im->first.print();
        for (unsigned int i=0; i<im->second.size(); i++) sss<<" "<<setw(6)<<im->second[i];
//      if (chnames.size() >1) { sss<<" *****"; for (unsigned int i=0; i<chnames.size();++i) sss<<" "<<chnames[i]; }
        sss<<std::endl;

    unsigned int idx = 0;
    while (idx < vectNamesCh.size()) {
      if (vectNamesCh[idx] == chamber.chamberLocationName && vectNamesPart[idx] == feb.localEtaPartition) break;
      idx++;
    }
    if (idx == vectNamesCh.size()) {

      vectRes.push_back(im->second);
      vectNamesCh.push_back(chamber.chamberLocationName);
      vectNamesPart.push_back(feb.localEtaPartition);
      vectNamesLink.push_back(im->first.print());
      vectSums.push_back(make_pair(idx,stat));
      vectMean.push_back(mean);
      vectRMS.push_back(sqrt(rms2/stat));
    }
  }

  sss <<endl<<endl<<"GOING TO WRITE: " << endl;
  sort(vectSums.begin(), vectSums.end(), lessVectSum() );
  for (vector<std::pair<unsigned int, unsigned int> >::const_iterator it = vectSums.begin();
       it != vectSums.end(); ++it) {

    unsigned int iindex = it->first;

    histo->Fill(vectMean[iindex]-3., vectRMS[iindex]);
    sss <<  vectNamesCh[iindex] <<" "
        << vectNamesPart[iindex]  <<" mean: "<<vectMean[iindex]<<" rms: "
        << vectRMS[iindex] << vectNamesLink[iindex];
    for (unsigned int i=0;  i< vectRes[iindex].size(); ++i) sss <<" "<<setw(6)<<vectRes[iindex][i];

    sss <<endl;

  }
  return sss.str();
}

