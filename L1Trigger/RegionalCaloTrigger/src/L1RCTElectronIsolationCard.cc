#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTElectronIsolationCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include <iostream>
#include <iomanip>

L1RCTElectronIsolationCard::L1RCTElectronIsolationCard(int crateNumber,
						       int cardNumber,
						       const L1RCTLookupTables* rctLookupTables) :
  crtNo(crateNumber),cardNo(cardNumber),
  rctLookupTables_(rctLookupTables),
  isoElectrons(2),nonIsoElectrons(2), regions(2)
{
  regions.push_back(L1RCTRegion());
  regions.push_back(L1RCTRegion());
}

L1RCTElectronIsolationCard::~L1RCTElectronIsolationCard()
{
  regions.clear();
}


void L1RCTElectronIsolationCard::fillElectronCandidates(){
  std::vector<unsigned short> region0Electrons = calcElectronCandidates(regions.at(0),0);
  std::vector<unsigned short> region1Electrons = calcElectronCandidates(regions.at(1),1);
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
L1RCTElectronIsolationCard::calcElectronCandidates(const L1RCTRegion& region, int regionNum){
  
  unsigned short nonIsoElectron = 0;
  unsigned short isoElectron = 0;
  
  //i is row and j is column
  for(int i = 0; i<4; i++){
    for(int j = 0; j<4; j++){

      unsigned short primaryEt = region.getEtIn7Bits(i,j);
      unsigned short primaryHE_FG = region.getHE_FGBit(i,j); 
      
      unsigned short northEt = region.getEtIn7Bits(i-1,  j);
      unsigned short southEt = region.getEtIn7Bits(i+1,  j);
      unsigned short westEt  = region.getEtIn7Bits(  i,j-1);
      unsigned short eastEt  = region.getEtIn7Bits(  i,j+1);
      unsigned short neEt    = region.getEtIn7Bits(i-1,j+1);
      unsigned short nwEt    = region.getEtIn7Bits(i-1,j-1);
      unsigned short seEt    = region.getEtIn7Bits(i+1,j+1);
      unsigned short swEt    = region.getEtIn7Bits(i+1,j-1);

      unsigned short northHE_FG = region.getHE_FGBit(i-1,  j);
      unsigned short southHE_FG = region.getHE_FGBit(i+1,  j);
      unsigned short westHE_FG  = region.getHE_FGBit(  i,j-1);
      unsigned short eastHE_FG  = region.getHE_FGBit(  i,j+1);
      unsigned short neHE_FG    = region.getHE_FGBit(i-1,j+1);
      unsigned short nwHE_FG    = region.getHE_FGBit(i-1,j-1);
      unsigned short seHE_FG    = region.getHE_FGBit(i+1,j+1);
      unsigned short swHE_FG    = region.getHE_FGBit(i+1,j-1);

      bool top = false;

      int nCrate = crateNumber();
      int nCard = cardNumber();
      int nRegion = regionNum;

      // top row of crate
      if (nCard == 0 || nCard == 2 || nCard == 4 || (nCard == 6 && nRegion == 0))
	{
	  top = true;
	}
      // bottom row of crate
      else if (nCard == 1 || nCard == 3 || nCard == 5 || (nCard == 6 && nRegion == 1))
	{} // top already false
      else
	{
	  std::cerr << "Error! EIC top assignment" << std::endl;  // this shouldn't happen!
	}

      // The following values are used for zeroing and determining whether or
      // not a tower is a "candidate".  The original primaryEt, northEt, neEt, etc.
      // are used to calculate vetoes and must not be zeroed.

      unsigned short primaryTowerEt = primaryEt;
      unsigned short northTowerEt = northEt;
      unsigned short southTowerEt = southEt;
      unsigned short eastTowerEt = eastEt;
      unsigned short westTowerEt = westEt;

      // In order to ensure proper selection of candidate tower and neighbor,
      // if two neighbor energies are equal, one is set to zero (in the
      // appropriate regions, those for which tp_lf bit is set to 0 in
      // Pam's JCCTest/EGWithShare.cc). 

      if (primaryEt > 0)  // this value should maybe be customizable?
	{
	  if (nCard != 6)  // all cards except 6
	    {
	      if(top && nCrate >= 9)  // top row of regions in positive-eta crate
		{
		  if(westTowerEt == eastTowerEt) westTowerEt=0;
		  if(southTowerEt == northTowerEt) southTowerEt=0;
		  if(southTowerEt == eastTowerEt) southTowerEt=0;
		  if(westTowerEt == northTowerEt) westTowerEt=0;
                }
              else if((!top) && nCrate < 9) // bottom row of regions in negative-eta crate
		{
		  if(eastTowerEt == westTowerEt) eastTowerEt=0;
		  if(northTowerEt == southTowerEt) northTowerEt=0;
		  if(northTowerEt == westTowerEt) northTowerEt=0;
		  if(eastTowerEt == southTowerEt) eastTowerEt=0;
		}
            }
          else  // card 6                                            
            {
	      // only +eta card 6 needs to have zeroing.  Pam sez.  
	      // -eta card 6 does what it's supposed to even w/o zeroing.
              if(nRegion == 0 && nCrate >=9)
                {
                  if(westTowerEt == eastTowerEt) westTowerEt=0;
                  if(southTowerEt == northTowerEt) southTowerEt=0;
                  if(southTowerEt == eastTowerEt) southTowerEt=0;
                  if(westTowerEt == northTowerEt) westTowerEt=0;
                }
	    }
	}
      
      // This section compares the energies in the primary tower with the
      // surrounding towers to determine whether or not the primary tower
      // should be considered a "candidate".  

      bool candidate = false;

      // for case where primary tower et greater than all neighbors -> candidate
      if (primaryEt > northEt && primaryEt > southEt && primaryEt > eastEt
	  && primaryEt > westEt && !primaryHE_FG)
	{
	  candidate = true;
	}

      // if primary et less than any neighbors (or HE_FG veto set) NOT a candidate!
      else if (primaryEt < northEt || primaryEt < southEt || primaryEt < eastEt
	  || primaryEt < westEt || primaryHE_FG)
	{} // candidate already false

      else // Case of primary tower et being equal to any of its neighbors.
	// This section determines which tower gets the candidate.
	// See AboutTP.pdf document, figure on p. 4, for clarification.
	// Zeroed values are used in this calculation.
	{
	  if (primaryEt > 0)
	    {
	      if (nCrate >=9)  // positive eta
		{
		  if (top) // top row of regions in crate.  tp_lf == 0
		    // priority order: east < south < north < west
		    {
		      if (westTowerEt == primaryTowerEt)
			candidate = true;
		      else if (northTowerEt == primaryTowerEt)
			candidate = false;
		      else if (southTowerEt == primaryTowerEt)
			candidate = true;
		      else if (eastTowerEt == primaryTowerEt)
			candidate = false;
		    }

		  else // bottom row of regions in crate.  tp_lf == 1
		    // priority order: west < north < south < east
		    {
		      if (eastTowerEt == primaryTowerEt)
			candidate = true;
		      else if (southTowerEt == primaryTowerEt)
			candidate = true;
		      else if (northTowerEt == primaryTowerEt)
			candidate = false;
		      else if (westTowerEt == primaryTowerEt)
			candidate = false;
		      if (nCard == 6)  // card 6. tp_lf == 1
			{
			  // priority order: east < north < south < west
			  if (westTowerEt == primaryTowerEt)
			    candidate = true;
			  else if (southTowerEt == primaryTowerEt)
			    candidate = true;
			  else if (northTowerEt == primaryTowerEt)
			    candidate = false;
			  else if (eastTowerEt == primaryTowerEt)
			    candidate = false;
			}
		    }
		}
	      else // negative eta
		{
		  if (top) // top row of regions in crate.  tp_lf == 1  
		    // priority order: east < south < north < west
		    {
		      if (westTowerEt == primaryTowerEt)
			candidate = true;
		      else if (northTowerEt == primaryTowerEt)
			candidate = true;
		      else if (southTowerEt == primaryTowerEt)
			candidate = false;
		      else if (eastTowerEt == primaryTowerEt)
			candidate = false;
		      if (nCard == 6)  // card 6.  tp_lf == 0
			// east < south < north < west
			{
			  if (westTowerEt == primaryTowerEt)
			    candidate = false;
			  else if (northTowerEt == primaryTowerEt)
			    candidate = false;
			  else if (southTowerEt == primaryTowerEt)
			    candidate = true;
			  else if (eastTowerEt == primaryTowerEt)
			    candidate = true;
			}
		    }
		  else // bottom row of regions.  tp_lf == 0
		    // west < north < south < east
		    {
		      if (eastTowerEt == primaryTowerEt)
			candidate = true;
		      else if (southTowerEt == primaryTowerEt)
			candidate = false;
		      else if (northTowerEt == primaryTowerEt)
			candidate = true;
		      else if (westTowerEt == primaryTowerEt)
			candidate = false;
		      
		      if (nCard == 6) // card 6.  tp_lf == 1
			// west < north < south < east
			{
			  if (eastTowerEt == primaryTowerEt)
			    candidate = true;
			  else if (southTowerEt == primaryTowerEt)
			    candidate = true;
			  else if (northTowerEt == primaryTowerEt)
			    candidate = false;
			  else if (westTowerEt == primaryTowerEt)
			    candidate = false;
			}
		    }
		}
	    }
	} // end of if (primary == neighbors)
      
      if (candidate) {
	
	// Either zeroed or non-zeroed set of values can be used here --
	// neighbor tower only zeroed if another neighbor tower of same
	// energy.  Max sum calculated from primary and only one neighbor
	// tower, and always one neighbor tower left over, so value of sum
	// is not affected.  Currently using non-zeroed.
	unsigned short candidateEt = calcMaxSum(primaryEt,northEt,southEt,
						eastEt,westEt);
	
	// neighbor HE_FG veto true if neighbor has HE_FG set
	bool neighborVeto = (nwHE_FG || northHE_FG || neHE_FG || westHE_FG ||
			     eastHE_FG || swHE_FG || southHE_FG || seHE_FG); 
	
	// threshold for five-tower corner quiet veto
	//int quietThreshold = 3;   // 3 - loose isolation 0 - very tight isolation
	//int quietThreshold = 7; // ECALGREN
	//int quietThreshold = 0; // HCALGREN
	unsigned quietThreshold = rctLookupTables_->rctParameters()->eicIsolationThreshold();
	
	bool nw = false;
	bool ne = false;
	bool sw = false;
	bool se = false;
	bool n = false;
	bool w = false;
	bool s = false;
	bool e = false;

	// individual neighbor vetoes set if neighbor is over threshold
	if (nwEt >= quietThreshold) nw = true;
	if (neEt >= quietThreshold) ne = true;
	if (swEt >= quietThreshold) sw = true;
	if (seEt >= quietThreshold) se = true;
	if (northEt >= quietThreshold) n = true;
	if (southEt >= quietThreshold) s = true;
	if (westEt >= quietThreshold) w = true;
	if (eastEt >= quietThreshold) e = true;

	// veto TRUE for each corner set if any individual tower in each set is over threshold
	bool nwC = (sw || w || nw || n || ne);
	bool neC = (nw || n || ne || e || se);
	bool seC = (ne || e || se || s || sw);
	bool swC = (se || s || sw || w || nw);

	// overall quiet veto TRUE only if NO corner sets are quiet 
	// (all are "loud") -> non-isolated
	bool quietVeto = (nwC && neC && seC && swC);

	// only isolated if both vetoes are false
	// Note: quietThreshold = 0 forces all candidates to be non-iso
	if(!(quietVeto || neighborVeto)){
	  if(candidateEt > isoElectron)
	    isoElectron = candidateEt;
	}
	// otherwise, non-isolated
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
  regions.at(0).print();

  std::cout << "IsoElectron Candidate " << isoElectrons.at(0) << std::endl;
  std::cout << "NonIsoElectron Candidate " << nonIsoElectrons.at(0) << std::endl << std::endl;

  std::cout << "Region 1 Information" << std::endl;
  regions.at(1).print();

  std::cout << "IsoElectron Candidate " << isoElectrons.at(1) << std::endl;
  std::cout << "NonIsoElectron Candidate " << nonIsoElectrons.at(1) << std::endl;
}
