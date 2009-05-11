/*************************************************************************
 *
 *   Program:  L1RCTFilter.cc
 *   Author:   K. Grogg, M. Weinberg
 *   Date:     1 May, 2007
 *   Revision: 25 July, 2007, 14 April, 2008 for new pattern tests
 *   Purpose:  Filter out desired events when running NCrateTest.
 *             Selected as true only those events that have a significant 
 *             amount of energy within a specified group of crates and eta. 
 *             
 *
 *************************************************************************/

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTFilter.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"


L1RCTFilter::L1RCTFilter(const edm::ParameterSet& ps) :
  pTMin(ps.getUntrackedParameter<double>("pTMin",0)),
  rctDigisLabel(ps.getParameter<edm::InputTag>("rctDigisLabel"))
  {
    vector<int> temp;
    temp.push_back(0);
    cardNumber=ps.getUntrackedParameter< vector<int> >("cardNumber",temp);
    cout << "filter!" << endl;
    cout << "cards " ;
    for(vector<int>::const_iterator num = cardNumber.begin();num!=cardNumber.end();++num)
      cout << (*num) << " "; 
    cout << endl;
  }
     
L1RCTFilter::~L1RCTFilter()
  {
  }
/************************************************************************
 *
 * Main class. Loops over all generated particles in the event. 
 * If particle is the specified type and has more than the min pt,
 * then loops over the selected crates.  If particle is in one of
 * the selected crates, then event is set to good. 
 * Finally, check for edge energy, if none, then event if still good.
 *
 ************************************************************************/ 
bool L1RCTFilter::filter(edm::Event& iEvent, edm::EventSetup const&)
{    
  //cout << "filter function!" << endl;
  bool goodEvent = false;
  static int badEvent = 0;
  static int numGoodEvents = 0;
 
  //FILTER BY RCT EM CANDS AND REGIONS
  goodEvent = false;
  //cout << "good event " << goodEvent << endl << flush;
  Handle<L1CaloEmCollection> rctEmCands;
  Handle<L1CaloRegionCollection> rctRegions;
  
  L1CaloEmCollection::const_iterator em;
  L1CaloRegionCollection::const_iterator rgn;
  
  iEvent.getByLabel(rctDigisLabel, rctEmCands);
  iEvent.getByLabel(rctDigisLabel, rctRegions);

  int cardCount[7] = {0};
  //static int countCard[7] = {0};
  for(em=rctEmCands->begin(); em!=rctEmCands->end(); em++){
    unsigned short card = (unsigned short) em->rctCard();
    unsigned short rank = (unsigned short) em->rank();
    //not right
    cout << "rankd " << rank << endl;
    if(rank>pTMin){
      //goodEvent = true;
       cardCount[card]++;
    }
  }
  //int count = numGoodEvents/countNumber;
  for(vector<int>::const_iterator num = cardNumber.begin();num!=cardNumber.end();++num)
    if(cardCount[(*num)]>0) //||cardCount[cardNumber-1]>0||cardCount[cardNumber-2]>0)
      goodEvent= true;

//   if(!goodEvent)
//     cout << "bad em\n";
  bool goodRegion = false;
  for (rgn=rctRegions->begin(); rgn!=rctRegions->end(); rgn++){
    unsigned short rank = (unsigned short) rgn->et();
    unsigned short card = (unsigned short) rgn->rctCard();
    for(vector<int>::const_iterator num = cardNumber.begin();num!=cardNumber.end();++num)
    if(rank>pTMin && (*num)==card)
      {
       //goodEvent = true; 
       goodRegion=true;
      }
  }
  //if(!goodRegion)
  //cout << "bad region\n";

  if(goodEvent == false) {badEvent++;}
  if(goodEvent) {numGoodEvents++;}
  
  if(numGoodEvents==65)
    cout << "num badEvents: " << badEvent << endl;
  
  return goodEvent;  
}

void L1RCTFilter::endJob()
{
}


