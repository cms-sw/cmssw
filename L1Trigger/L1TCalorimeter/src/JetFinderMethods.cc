// JetFinderMethods.cc
// Author: Alex Barbieri
//
// This file should contain the different algorithms used to find jets.
// Currently the standard is the sliding window method, used by both
// HI and PP.

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include <vector>

namespace l1t {

  int deltaGctPhi(const CaloRegion & region, const CaloRegion & neighbor)
  {
    int phi1 = region.hwPhi();
    int phi2 = neighbor.hwPhi();
    int diff = phi1 - phi2;
    if (std::abs(phi1 - phi2) == L1CaloRegionDetId::N_PHI-1) { //18 regions in phi
      diff = -diff/std::abs(diff);
    }
    return diff;
  }

  // turn each central region into a jet
  void passThroughJets(const std::vector<l1t::CaloRegion> * regions,
		       std::vector<l1t::Jet> * uncalibjets)
  {
    for(std::vector<CaloRegion>::const_iterator region = regions->begin(); region != regions->end(); region++) {
      int jetQual = 0;
      if( region->hwEta() < 4 || region->hwEta() > 17)
	jetQual = 2;
      int jetET = region->hwPt();
      int jetEta = region->hwEta();
      int jetPhi = region->hwPhi();

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > jetLorentz(0,0,0,0);
      l1t::Jet theJet(*&jetLorentz, jetET, jetEta, jetPhi, jetQual);
      uncalibjets->push_back(theJet);
    }
  }

  void slidingWindowJetFinder(const int jetSeedThreshold,
			      const std::vector<l1t::CaloRegion> * regions,
			      std::vector<l1t::Jet> * uncalibjets)
  {
    // std::cout << "Jet Seed: " << jetSeedThreshold << std::endl;
    for(std::vector<CaloRegion>::const_iterator region = regions->begin(); region != regions->end(); region++) {
      int regionET = region->hwPt(); //regionPhysicalEt(*region);
      if (regionET  <= jetSeedThreshold) continue;
      int neighborN_et = 0;
      int neighborS_et = 0;
      int neighborE_et = 0;
      int neighborW_et = 0;
      int neighborNE_et = 0;
      int neighborSW_et = 0;
      int neighborNW_et = 0;
      int neighborSE_et = 0;
      unsigned int nNeighbors = 0;
      for(std::vector<CaloRegion>::const_iterator neighbor = regions->begin(); neighbor != regions->end(); neighbor++) {
	int neighborET = neighbor->hwPt(); //regionPhysicalEt(*neighbor);
	if(deltaGctPhi(*region, *neighbor) == 1 &&
	   (region->hwEta()    ) == neighbor->hwEta()) {
	  neighborN_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta()    ) == neighbor->hwEta()) {
	  neighborS_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 0 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 0 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 1 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborNE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborSW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 1 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborNW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborSE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
      }
      if(regionET > neighborN_et &&
	 regionET > neighborNW_et &&
	 regionET > neighborW_et &&
	 regionET > neighborSW_et &&
	 regionET >= neighborNE_et &&
	 regionET >= neighborE_et &&
	 regionET >= neighborSE_et &&
	 regionET >= neighborS_et) {
	unsigned int jetET = regionET +
	  neighborN_et + neighborS_et + neighborE_et + neighborW_et +
	  neighborNE_et + neighborSW_et + neighborSE_et + neighborNW_et;

	int jetPhi = region->hwPhi();
	int jetEta = region->hwEta();

	bool neighborCheck = (nNeighbors == 8);
	// On the eta edge we only expect 5 neighbors
	if (!neighborCheck && (jetEta == 0 || jetEta == 21) && nNeighbors == 5)
	  neighborCheck = true;

	if (!neighborCheck) {
	  std::cout << "phi: " << jetPhi << " eta: " << jetEta << " n: " << nNeighbors << std::endl;
	  assert(false);
	}

	//first iteration, eta cut defines forward
	//const bool forward = (jetEta <= 4 || jetEta >= 17);
	const bool forward = (jetEta < 4 || jetEta > 17);
	int jetQual = 0;
	if(forward)
	  jetQual |= 0x2;

	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > jetLorentz(0,0,0,0);
	l1t::Jet theJet(*&jetLorentz, jetET, jetEta, jetPhi, jetQual);
	//l1t::Jet theJet(0, jetET, jetEta, jetPhi);

	uncalibjets->push_back(theJet);
      }
    }
  }

  void TwelveByTwelveFinder(const int jetSeedThreshold,
			      const std::vector<l1t::CaloRegion> * regions,
			      std::vector<l1t::Jet> * uncalibjets)
  {
    // std::cout << "Jet Seed: " << jetSeedThreshold << std::endl;
    for(std::vector<CaloRegion>::const_iterator region = regions->begin(); region != regions->end(); region++) {
      int regionET = region->hwPt(); //regionPhysicalEt(*region);
      if (regionET  < jetSeedThreshold) continue;
      int neighborN_et = 0;
      int neighborS_et = 0;
      int neighborE_et = 0;
      int neighborW_et = 0;
      int neighborNE_et = 0;
      int neighborSW_et = 0;
      int neighborNW_et = 0;
      int neighborSE_et = 0;
      unsigned int nNeighbors = 0;
      for(std::vector<CaloRegion>::const_iterator neighbor = regions->begin(); neighbor != regions->end(); neighbor++) {
	int neighborET = neighbor->hwPt(); //regionPhysicalEt(*neighbor);
	if(deltaGctPhi(*region, *neighbor) == 1 &&
	   (region->hwEta()    ) == neighbor->hwEta()) {
	  neighborN_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta()    ) == neighbor->hwEta()) {
	  neighborS_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 0 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 0 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 1 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborNE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborSW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 1 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborNW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborSE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
      }
      unsigned int jetET = regionET +
	neighborN_et + neighborS_et + neighborE_et + neighborW_et +
	neighborNE_et + neighborSW_et + neighborSE_et + neighborNW_et;

      int jetPhi = region->hwPhi();
      int jetEta = region->hwEta();

      bool neighborCheck = (nNeighbors == 8);
      // On the eta edge we only expect 5 neighbors
      if (!neighborCheck && (jetEta == 0 || jetEta == 21) && nNeighbors == 5)
	neighborCheck = true;

      if (!neighborCheck) {
	std::cout << "phi: " << jetPhi << " eta: " << jetEta << " n: " << nNeighbors << std::endl;
	assert(false);
      }

      //first iteration, eta cut defines forward
      //const bool forward = (jetEta <= 4 || jetEta >= 17);
      const bool forward = (jetEta < 4 || jetEta > 17);
      int jetQual = 0;
      if(forward)
	jetQual |= 0x2;

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > jetLorentz(0,0,0,0);
      l1t::Jet theJet(*&jetLorentz, jetET, jetEta, jetPhi, jetQual);
      //l1t::Jet theJet(0, jetET, jetEta, jetPhi);

      uncalibjets->push_back(theJet);
    }
  }

  void TwoByTwoFinder(const std::vector<l1t::CaloRegion> * regions,
		      std::vector<l1t::Jet> * uncalibjets)
  {
    for(std::vector<CaloRegion>::const_iterator region = regions->begin(); region != regions->end(); region++) {
      int regionET = region->hwPt();
      int neighborN_et = 0;
      int neighborS_et = 0;
      int neighborE_et = 0;
      int neighborW_et = 0;
      int neighborNE_et = 0;
      int neighborSW_et = 0;
      int neighborNW_et = 0;
      int neighborSE_et = 0;
      unsigned int nNeighbors = 0;
      for(std::vector<CaloRegion>::const_iterator neighbor = regions->begin(); neighbor != regions->end(); neighbor++) {
	int neighborET = neighbor->hwPt();
	if(deltaGctPhi(*region, *neighbor) == 1 &&
	   (region->hwEta()    ) == neighbor->hwEta()) {
	  neighborN_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta()    ) == neighbor->hwEta()) {
	  neighborS_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 0 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 0 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 1 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborNE_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborSW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == 1 &&
		(region->hwEta() - 1) == neighbor->hwEta()) {
	  neighborNW_et = neighborET;
	  nNeighbors++;
	  continue;
	}
	else if(deltaGctPhi(*region, *neighbor) == -1 &&
		(region->hwEta() + 1) == neighbor->hwEta()) {
	  neighborSE_et = neighborET;
	  nNeighbors++;
	  continue;
	}

      }
      if(regionET > neighborN_et &&
	 regionET > neighborNW_et &&
	 regionET > neighborW_et &&
	 regionET > neighborSW_et &&
	 regionET >= neighborNE_et &&
	 regionET >= neighborE_et &&
	 regionET >= neighborSE_et &&
	 regionET >= neighborS_et) {

	// use the highest-pT 2x2 jet inside this 3x3
	unsigned int jetET_NW;
	unsigned int jetET_NE;
	unsigned int jetET_SW;
	unsigned int jetET_SE;

	jetET_NW = regionET + neighborW_et + neighborNW_et + neighborN_et;
	jetET_NE = regionET + neighborE_et + neighborNE_et + neighborN_et;
	jetET_SW = regionET + neighborS_et + neighborSW_et + neighborW_et;
	jetET_SE = regionET + neighborS_et + neighborSE_et + neighborE_et;

	unsigned int jetET = std::max(jetET_NW, jetET_NE);
	jetET = std::max(jetET, jetET_SW);
	jetET = std::max(jetET, jetET_SE);

	int jetPhi = region->hwPhi();
	int jetEta = region->hwEta();

	bool neighborCheck = (nNeighbors == 8);
	// On the eta edge we only expect 5 neighbor
	if (!neighborCheck && (jetEta == 0 || jetEta == 21) && nNeighbors == 5)
	  neighborCheck = true;

	if (!neighborCheck) {
	  std::cout << "phi: " << jetPhi << " eta: " << jetEta << " n: " << nNeighbors << std::endl;
	  assert(false);
	}

	//first iteration, eta cut defines forward
	const bool forward = (jetEta < 4 || jetEta > 17);
	int jetQual = 0;
	if(forward)
	  jetQual |= 0x2;

	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > jetLorentz(0,0,0,0);
	l1t::Jet theJet(*&jetLorentz, jetET, jetEta, jetPhi, jetQual);
	uncalibjets->push_back(theJet);
      }
    }
  }
}
