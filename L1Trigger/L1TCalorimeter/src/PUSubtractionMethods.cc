// JetFinderMethods.cc
// Author: Alex Barbieri
//
// This file should contain the different algorithms used to find jets.
// Currently the standard is the sliding window method, used by both
// HI and PP.

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include <vector>

namespace l1t {

  void HICaloRingSubtraction(const std::vector<l1t::CaloRegion> & regions,
			     std::vector<l1t::CaloRegion> *subRegions)
  {
    int puLevelHI[L1CaloRegionDetId::N_ETA];
    double r_puLevelHI[L1CaloRegionDetId::N_ETA];
    int etaCount[L1CaloRegionDetId::N_ETA];
    for(unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i)
    {
      puLevelHI[i] = 0;
      r_puLevelHI[i] = 0.0;
      etaCount[i] = 0;
    }

    for(std::vector<CaloRegion>::const_iterator region = regions.begin();
	region != regions.end(); region++){
      r_puLevelHI[region->hwEta()] += region->hwPt();
      etaCount[region->hwEta()]++;
    }

    for(unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i)
    {
      puLevelHI[i] = floor(r_puLevelHI[i]/etaCount[i] + 0.5);
    }

    for(std::vector<CaloRegion>::const_iterator region = regions.begin(); region!= regions.end(); region++){
      int subPt = std::max(0, region->hwPt() - puLevelHI[region->hwEta()]);
      int subEta = region->hwEta();
      int subPhi = region->hwPhi();

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *lorentz =
	new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

      CaloRegion newSubRegion(*lorentz, 0, 0, subPt, subEta, subPhi, 0, 0, 0);
      subRegions->push_back(newSubRegion);
    }
  }

}
