// PUSubtractionMethods.cc
// Authors: Alex Barbieri
//          Kalanand Mishra, Fermilab
//          Inga Bucinskaite, UIC
//
// This file should contain the different algorithms used to perform PU, UE subtraction.

//#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "TMath.h"

//#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include <vector>

namespace l1t {

  /// --------------- For heavy ion -------------------------------------
  void HICaloRingSubtraction(const std::vector<l1t::CaloRegion> &regions,
                             std::vector<l1t::CaloRegion> *subRegions,
                             CaloParamsHelper const *params) {
    int puLevelHI[L1CaloRegionDetId::N_ETA];

    for (unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i) {
      puLevelHI[i] = 0;
    }

    for (std::vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {
      puLevelHI[region->hwEta()] += region->hwPt();
    }

    for (unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i) {
      //puLevelHI[i] = floor(((double)puLevelHI[i] / (double)L1CaloRegionDetId::N_PHI)+0.5);
      puLevelHI[i] = (puLevelHI[i] + 9) * 455 / (1 << 13);  // approx equals X/18 +0.5
    }

    for (std::vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {
      int subPt = std::max(0, region->hwPt() - puLevelHI[region->hwEta()]);
      int subEta = region->hwEta();
      int subPhi = region->hwPhi();

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0, 0, 0, 0);

      CaloRegion newSubRegion(
          *&ldummy, 0, 0, subPt, subEta, subPhi, region->hwQual(), region->hwEtEm(), region->hwEtHad());
      subRegions->push_back(newSubRegion);
    }
  }

  void simpleHWSubtraction(const std::vector<l1t::CaloRegion> &regions, std::vector<l1t::CaloRegion> *subRegions) {
    for (std::vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {
      int subEta = region->hwEta();
      int subPhi = region->hwPhi();
      int subPt = region->hwPt();

      if (subPt != (2 << 10) - 1)
        subPt = subPt - (10 + subEta);  // arbitrary value chosen in meeting
      if (subPt < 0)
        subPt = 0;
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0, 0, 0, 0);

      CaloRegion newSubRegion(
          *&ldummy, 0, 0, subPt, subEta, subPhi, region->hwQual(), region->hwEtEm(), region->hwEtHad());
      subRegions->push_back(newSubRegion);
    }
  }

  /// --------- New region correction (PUsub, no response correction at the moment) -----------

  void RegionCorrection(const std::vector<l1t::CaloRegion> &regions,
                        std::vector<l1t::CaloRegion> *subRegions,
                        CaloParamsHelper const *params) {
    std::string regionPUSType = params->regionPUSType();

    if (regionPUSType == "None") {
      for (std::vector<CaloRegion>::const_iterator notCorrectedRegion = regions.begin();
           notCorrectedRegion != regions.end();
           notCorrectedRegion++) {
        const CaloRegion &newSubRegion = *notCorrectedRegion;
        subRegions->push_back(newSubRegion);
      }
    }

    if (regionPUSType == "HICaloRingSub") {
      HICaloRingSubtraction(regions, subRegions, params);
    }

    if (regionPUSType == "PUM0") {
      int puMult = 0;

      // ------------ This calulates PUM0 ------------------
      for (std::vector<CaloRegion>::const_iterator notCorrectedRegion = regions.begin();
           notCorrectedRegion != regions.end();
           notCorrectedRegion++) {
        int regionET = notCorrectedRegion->hwPt();
        if (regionET > 0) {
          puMult++;
        }
      }
      int pumbin = (int)puMult / 22;
      if (pumbin == 18)
        pumbin = 17;  // if puMult = 396 exactly there is an overflow

      for (std::vector<CaloRegion>::const_iterator notCorrectedRegion = regions.begin();
           notCorrectedRegion != regions.end();
           notCorrectedRegion++) {
        int regionET = notCorrectedRegion->hwPt();
        int regionEta = notCorrectedRegion->hwEta();
        int regionPhi = notCorrectedRegion->hwPhi();

        //int puSub = ceil(regionPUSParams[18*regionEta+pumbin]*2);
        int puSub = params->regionPUSValue(pumbin, regionEta);
        // The values in regionSubtraction are MULTIPLIED by
        // RegionLSB=.5 (physicalRegionEt), so to get back unmultiplied
        // regionSubtraction we want to multiply the number by 2
        // (aka divide by LSB).

        int regionEtCorr = std::max(0, regionET - puSub);
        if (regionET == 1023)
          regionEtCorr = 1023;  // do not subtract overflow regions
        if ((regionET == 255) && (regionEta < 4 || regionEta > 17))
          regionEtCorr = 255;

        ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > lorentz(0, 0, 0, 0);
        CaloRegion newSubRegion(*&lorentz,
                                0,
                                0,
                                regionEtCorr,
                                regionEta,
                                regionPhi,
                                notCorrectedRegion->hwQual(),
                                notCorrectedRegion->hwEtEm(),
                                notCorrectedRegion->hwEtHad());
        subRegions->push_back(newSubRegion);
      }
    }
  }
}  // namespace l1t
