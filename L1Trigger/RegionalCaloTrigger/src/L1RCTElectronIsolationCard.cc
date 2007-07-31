#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTElectronIsolationCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include <iostream>
#include <iomanip>

L1RCTElectronIsolationCard::L1RCTElectronIsolationCard(int crateNumber,
						       int cardNumber,
						       const L1RCTLookupTables* rctLookupTables) :
  crtNo(crateNumber),cardNo(cardNumber),
  rctLookupTables_(rctLookupTables),
  isoElectrons(2),nonIsoElectrons(2), regions(2)
{
  regions.at(0) = new L1RCTRegion();
  regions.at(1) = new L1RCTRegion();
}

L1RCTElectronIsolationCard::~L1RCTElectronIsolationCard(){}


void L1RCTElectronIsolationCard::fillElectronCandidates(){
  std::vector<unsigned short> region0Electrons = calcElectronCandidates(regions.at(0));
  std::vector<unsigned short> region1Electrons = calcElectronCandidates(regions.at(1));
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
std::vector<unsigned short>
L1RCTElectronIsolationCard::calcElectronCandidates(L1RCTRegion* region){
  
  unsigned short nonIsoElectron = 0;
  unsigned short isoElectron = 0;
  
  //i is row and j is column
  for(int i = 0; i<4; i++){
    for(int j = 0; j<4; j++){

      unsigned short primaryEt = region->getEtIn7Bits(i,j);
      unsigned short primaryHE_FG = region->getHE_FGBit(i,j); 
      
      unsigned short northEt = region->getEtIn7Bits(i-1,  j);
      unsigned short southEt = region->getEtIn7Bits(i+1,  j);
      unsigned short westEt  = region->getEtIn7Bits(  i,j-1);
      unsigned short eastEt  = region->getEtIn7Bits(  i,j+1);
      unsigned short neEt    = region->getEtIn7Bits(i-1,j+1);
      unsigned short nwEt    = region->getEtIn7Bits(i-1,j-1);
      unsigned short seEt    = region->getEtIn7Bits(i+1,j+1);
      unsigned short swEt    = region->getEtIn7Bits(i+1,j-1);

      unsigned short northHE_FG = region->getHE_FGBit(i-1,  j);
      unsigned short southHE_FG = region->getHE_FGBit(i+1,  j);
      unsigned short westHE_FG  = region->getHE_FGBit(  i,j-1);
      unsigned short eastHE_FG  = region->getHE_FGBit(  i,j+1);
      unsigned short neHE_FG    = region->getHE_FGBit(i-1,j+1);
      unsigned short nwHE_FG    = region->getHE_FGBit(i-1,j-1);
      unsigned short seHE_FG    = region->getHE_FGBit(i+1,j+1);
      unsigned short swHE_FG    = region->getHE_FGBit(i+1,j-1);

      if(primaryEt > northEt && primaryEt >= southEt && primaryEt > westEt
	 && primaryEt >= eastEt && !primaryHE_FG){
	
	unsigned short candidateEt = calcMaxSum(primaryEt,northEt,southEt,
						eastEt,westEt);

	bool neighborVeto = (nwHE_FG || northHE_FG || neHE_FG || westHE_FG ||
			     eastHE_FG || swHE_FG || southHE_FG || seHE_FG); 
	
	int quietThreshold = 3;   // 3 - loose isolation 0 - very tight isolation
	
	bool nw = false;
	bool ne = false;
	bool sw = false;
	bool se = false;
	bool n = false;
	bool w = false;
	bool s = false;
	bool e = false;

	if (nwEt > quietThreshold) nw = true;
	if (neEt > quietThreshold) ne = true;
	if (swEt > quietThreshold) sw = true;
	if (seEt > quietThreshold) se = true;
	if (northEt > quietThreshold) n = true;
	if (southEt > quietThreshold) s = true;
	if (westEt > quietThreshold) w = true;
	if (eastEt > quietThreshold) e = true;

	bool nwC = (sw || w || nw || n || ne);
	bool neC = (nw || n || ne || e || se);
	bool seC = (ne || e || se || s || sw);
	bool swC = (se || s || sw || w || nw);

	bool quietVeto = (nwC && neC && seC && swC);

	if(!(quietVeto || neighborVeto)){
	  if(candidateEt > isoElectron)
	    isoElectron = candidateEt;
	}
	else if(candidateEt > nonIsoElectron)
	  nonIsoElectron = candidateEt;

      }
    }
  }

  
  std::vector<unsigned short> candidates;
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
  std::cout << "Electron isolation card " << cardNo << std::endl;
  std::cout << "Region 0 Information" << std::endl;
  regions.at(0)->print();

  std::cout << "IsoElectron Candidate " << isoElectrons.at(0) << std::endl;
  std::cout << "NonIsoElectron Candidate " << nonIsoElectrons.at(0) << std::endl << std::endl;

  std::cout << "Region 1 Information" << std::endl;
  regions.at(1)->print();

  std::cout << "IsoElectron Candidate " << isoElectrons.at(1) << std::endl;
  std::cout << "NonIsoElectron Candidate " << nonIsoElectrons.at(1) << std::endl;
}
