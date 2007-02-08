#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTElectronIsolationCard.h"

L1RCTElectronIsolationCard::L1RCTElectronIsolationCard(int crateNumber,
						       int cardNumber) :
  crtNo(crateNumber),cardNo(cardNumber),isoElectrons(2),nonIsoElectrons(2),
  regions(2)
{
  regions.at(0) = new L1RCTRegion();
  regions.at(1) = new L1RCTRegion();
}

L1RCTElectronIsolationCard::~L1RCTElectronIsolationCard(){}


void L1RCTElectronIsolationCard::fillElectronCandidates(){
  vector<unsigned short> region0Electrons = calcElectronCandidates(regions.at(0));
  vector<unsigned short> region1Electrons = calcElectronCandidates(regions.at(1));
  isoElectrons.at(0) = region0Electrons.at(0);
  isoElectrons.at(1) = region1Electrons.at(0);
  nonIsoElectrons.at(0) = region0Electrons.at(1);
  nonIsoElectrons.at(1) = region1Electrons.at(1);
}


//This method is the bulk of this class.  It finds the electrons given a pointer to
//a region.  It will return the largest nonIsoElectron candidate and the largest
//isoElectron candidate.  A deposit is an electron candidate if the h/e||fg bit is
//not on and it is higher energy than it's direct four neighbors.
//An electron candidate is *always* a non-isolated electron.
//If it also passes the neighbor cuts then it is an isolated electron as well.
vector<unsigned short>
L1RCTElectronIsolationCard::calcElectronCandidates(L1RCTRegion* region){
  
  unsigned short nonIsoElectron = 0;
  unsigned short isoElectron = 0;
  
  unsigned short primaryEt;
  unsigned short primaryHE_FG;
  unsigned short nwEt;
  unsigned short nwHE_FG;
  unsigned short neEt;
  unsigned short neHE_FG;
  unsigned short swEt;
  unsigned short swHE_FG;
  unsigned short seEt;
  unsigned short seHE_FG;
  unsigned short northEt;
  unsigned short northHE_FG;
  unsigned short southEt;
  unsigned short southHE_FG;
  unsigned short eastEt;
  unsigned short eastHE_FG;
  unsigned short westEt;
  unsigned short westHE_FG;
  //i is row and j is column
  for(int i = 0; i<4; i++){
    for(int j = 0; j<4; j++){
      primaryEt = region->getEtIn7Bits(i,j);
      primaryHE_FG = region->getHE_FGBit(i,j); 
      
      northEt = region->getEtIn7Bits(i-1,j);
      northHE_FG = region->getHE_FGBit(i-1,j);
      southEt = region->getEtIn7Bits(i+1,j);
      southHE_FG = region->getHE_FGBit(i+1,j);
      westEt = region->getEtIn7Bits(i,j-1);
      westHE_FG = region->getHE_FGBit(i,j-1);
      eastEt = region->getEtIn7Bits(i,j+1);
      eastHE_FG = region->getHE_FGBit(i,j+1);
      neEt = region->getEtIn7Bits(i-1,j+1);
      neHE_FG = region->getHE_FGBit(i-1,j+1);
      nwEt = region->getEtIn7Bits(i-1,j-1);
      nwHE_FG = region->getHE_FGBit(i-1,j-1);
      seEt = region->getEtIn7Bits(i+1,j+1);
      seHE_FG = region->getHE_FGBit(i+1,j+1);
      swEt = region->getEtIn7Bits(i+1,j-1);
      swHE_FG = region->getHE_FGBit(i+1,j-1);

      if(primaryEt > northEt && primaryEt >= southEt && primaryEt > westEt
	 && primaryEt >= eastEt && !primaryHE_FG){
	
	unsigned short candidateEt = calcMaxSum(primaryEt,northEt,southEt,
						eastEt,westEt);

	

	bool neighborVeto = (nwHE_FG || northHE_FG || neHE_FG || westHE_FG ||
			     eastHE_FG || swHE_FG || southHE_FG || seHE_FG); 
	
	//bool neighborVeto = false;
	//veto threshold
	int quietThreshold = 3;
	bool quietVeto = (nwEt > quietThreshold || neEt > quietThreshold || 
			  swEt > quietThreshold || seEt > quietThreshold);
	
	//bool quietVeto = false;
	
	if(!(quietVeto || neighborVeto)){
	  if(candidateEt > isoElectron)
	    isoElectron = candidateEt;
	}
	else if(candidateEt > nonIsoElectron)
	  nonIsoElectron = candidateEt;
      }
    }
  }

  
  vector<unsigned short> candidates;
  unsigned short fullIsoElectron = isoElectron*16 + cardNo*2;  // leaves room for last bit -- region number, added in Crate.cc
  candidates.push_back(fullIsoElectron);
  unsigned short fullNonIsoElectron = nonIsoElectron*16 + cardNo*2;  // leaves room for region info in last bit
  candidates.push_back(fullNonIsoElectron);

  return candidates;
}

unsigned short 
L1RCTElectronIsolationCard::calcMaxSum(unsigned short primaryEt,unsigned short northEt,
					unsigned short southEt,unsigned short eastEt,
					unsigned short westEt){
  unsigned short cardinals[4] = {northEt,southEt,eastEt,westEt};
  unsigned short max = 0;
  for(int i = 0; i<4;i++){
    unsigned short test = primaryEt+cardinals[i];
    if(test > max)
      max = test;
  }
  return max;
}

void L1RCTElectronIsolationCard::print() {
  cout << "Electron isolation card " << cardNo << endl;
  cout << "Region 0 Information" << endl;
  regions.at(0)->print();

  cout << "IsoElectron Candidate " << isoElectrons.at(0) << endl;
  cout << "NonIsoElectron Candidate " << nonIsoElectrons.at(0) << endl << endl;

  cout << "Region 1 Information" << endl;
  regions.at(1)->print();

  cout << "IsoElectron Candidate " << isoElectrons.at(1) << endl;
  cout << "NonIsoElectron Candidate " << nonIsoElectrons.at(1) << endl;
}
