#include "DQM/RPCMonitorClient/interface/RPCLinkSynchroHistoMaker.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include <sstream>
#include "TH1F.h"
#include "TH2F.h"

using namespace std;

bool RPCLinkSynchroHistoMaker::LessVectStatSum::operator() (const RPCLinkSynchroHistoMaker::LinkStat & o1, const RPCLinkSynchroHistoMaker::LinkStat & o2) { return o1.sum < o2.sum; }
bool RPCLinkSynchroHistoMaker::LessVectStatMean::operator() (const RPCLinkSynchroHistoMaker::LinkStat & o1, const RPCLinkSynchroHistoMaker::LinkStat & o2) { return o1.mean < o2.mean; }


RPCLinkSynchroHistoMaker::RPCLinkSynchroHistoMaker(
      const RPCRawSynchro & s,const RPCReadOutMapping* rm)
    : theRawSynchro(s), theCabling(rm), theUpdated(false)
{}

string RPCLinkSynchroHistoMaker::LinkStat::print() const
{
  ostringstream str;
  str<<nameChamber<<" "<<namePart<<" mean: "<<mean<<" rms: "<<rms<<nameLink<<" counts: "; 
  for (vector<int>::const_iterator it=vectStat.begin();it!=vectStat.end();++it) str<<" "<<(*it);
  return str.str();
}

void RPCLinkSynchroHistoMaker::makeLinkStats()
{
  if (!theCabling) return;
  if (theUpdated) return; else theUpdated=true;
  typedef RPCRawSynchro::LBCountMap::const_iterator IT;
  for (IT im = theRawSynchro.theSynchroCounts.begin(); im != theRawSynchro.theSynchroCounts.end(); ++im) {
    const LinkBoardSpec* linkBoard = theCabling->location(im->first);
    if (linkBoard==0) continue;
    const ChamberLocationSpec & chamber = linkBoard->febs().front().chamber();
    const FebLocationSpec & feb =  linkBoard->febs().front().feb();
    unsigned int idx = 0;
    while (idx < theLinkStat.size()) {
      if (theLinkStat[idx].nameChamber==chamber.chamberLocationName && theLinkStat[idx].namePart == feb.localEtaPartition) break;
      idx++;
    }
    if (idx== theLinkStat.size()) {
      LinkStat linkStat;
      linkStat.nameLink=im->first.print();
      linkStat.nameChamber=chamber.chamberLocationName;
      linkStat.namePart=feb.localEtaPartition;
      linkStat.vectStat=im->second.counts();
      linkStat.mean=im->second.mean();
      linkStat.sum=im->second.sum();
      linkStat.rms=im->second.rms();
      theLinkStat.push_back(linkStat);
    }
  }
}

string RPCLinkSynchroHistoMaker::dumpDelays() 
{  
  makeLinkStats();
  sort(theLinkStat.begin(), theLinkStat.end(), LessVectStatSum() );
  std::ostringstream str;
  str<<endl<<endl<<"GOING TO WRITE: " << endl;
  for (std::vector<LinkStat>::const_iterator it=theLinkStat.begin(); it != theLinkStat.end(); ++it) str<<it->print()<<endl;
  return str.str();
}

void RPCLinkSynchroHistoMaker::fillDelaySpreadHisto(TH2F* histo)
{
  makeLinkStats();
  for (std::vector<LinkStat>::const_iterator it=theLinkStat.begin(); it != theLinkStat.end(); ++it)histo->Fill(it->mean-3.,it->rms);
}

//void RPCLinkSynchroHistoMaker::fillLinksBadSynchro(TH2F* histo)
//{
//  makeLinkStats();
//  sort( theLinkStat.begin(), theLinkStat.end(), LessVectStatMean() );
//  for (unsigned int i=1; i<6; ++i) {
//    if( i > theLinkStat.size() ) break;
//    histo->GetYaxis()->SetBinLabel(i,theLinkStat[i-1].nameChamber.c_str());
//    for(unsigned int j=0; j<theLinkStat[i-1].vectStat.size();++j)histo->SetBinContent(j+1,i,theLinkStat[i-1].vectStat[j]);
//  }
//}
//
//void RPCLinkSynchroHistoMaker::fillLinksLowStat(TH2F* histo)
//{
//  makeLinkStats();
//  sort(theLinkStat.begin(), theLinkStat.end(), LessVectStatSum() );
//}
//
//void RPCLinkSynchroHistoMaker::fillLinksMostNoisy(TH2F* histo)
//{
//  makeLinkStats();
//  sort(theLinkStat.begin(), theLinkStat.end(), LessVectStatSum() );
//}

