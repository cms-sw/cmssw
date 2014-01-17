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

  void slidingWindowJetFinder(const std::vector<l1t::CaloRegion> * regions,
			      std::vector<l1t::Jet> * jets)
  {
    for(std::vector<CaloRegion>::const_iterator region = regions->begin(); region != regions->end(); region++) {
      double regionET = region->hwPt(); //regionPhysicalEt(*region);
      double neighborN_et = 0;
      double neighborS_et = 0;
      double neighborE_et = 0;
      double neighborW_et = 0;
      double neighborNE_et = 0;
      double neighborSW_et = 0;
      double neighborNW_et = 0;
      double neighborSE_et = 0;
      unsigned int nNeighbors = 0;
      for(std::vector<CaloRegion>::const_iterator neighbor = regions->begin(); neighbor != regions->end(); neighbor++) {
	double neighborET = neighbor->hwPt(); //regionPhysicalEt(*neighbor);
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
	/*
	  int jetPhi = region->hwPhi() * 4 +
	  ( - 2 * (neighborS_et + neighborSE_et + neighborSW_et)
	  + 2 * (neighborN_et + neighborNE_et + neighborNW_et) ) / jetET;
	  if(jetPhi < 0) {

	  }
	  else if(jetPhi >= ((int) N_JET_PHI)) {
	  jetPhi -= N_JET_PHI;
	  }
	  int jetEta = region->hwEta() * 4 +
	  ( - 2 * (neighborW_et + neighborNW_et + neighborSW_et)
	  + 2 * (neighborE_et + neighborNE_et + neighborSE_et) ) / jetET;
	  if(jetEta < 0) jetEta = 0;
	  if(jetEta >= ((int) N_JET_ETA)) jetEta = N_JET_ETA - 1;
	*/
	// Temporarily use the region granularity -- we will try to improve as above when code is debugged
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

	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *jetLorentz =
	  new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
	l1t::Jet theJet(*jetLorentz, jetET, jetEta, jetPhi);
	//l1t::Jet theJet(0, jetET, jetEta, jetPhi);

	jets->push_back(theJet);
      }
    }
  }
}
