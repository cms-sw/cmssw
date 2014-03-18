/// \class l1t::Stage1Layer2TauAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Kalanand Mishra - Fermilab
///


#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2TauAlgorithmImp.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"


void l1t::Stage1Layer2TauAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & clusters, const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::Tau> * taus) {

   tauSeed = 0;

   for(CaloRegionBxCollection::const_iterator region = regions.begin();
       region != regions.end(); region++) {

      double regionEt = region->hwPt(); //regionPhysicalEt(*region);
      if(regionEt < tauSeed) continue;
            
      double associatedSecondRegionEt = 0;
      double associatedThirdRegionEt = 0;
            
      findAnnulusInfo(region->hwEta(), region->hwPhi(),
                      regions, &associatedSecondRegionEt, 
                      &associatedThirdRegionEt);
            

      double tauEt=regionEt;
      if(associatedSecondRegionEt>tauSeed) tauEt +=associatedSecondRegionEt;


      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *tauLorentz =
         new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

      l1t::Tau theTau(*tauLorentz, tauEt, region->hwEta(), region->phi());
      taus->push_back(theTau);
   }

   //the taus should be sorted, highest pT first.
   // do not truncate the tau list, GT converter handles that
   auto comp = [&](l1t::Tau i, l1t::Tau j)-> bool {
        return (i.hwPt() < j.hwPt() ); 
    };

   std::sort(taus->begin(), taus->end(), comp);
   std::reverse(taus->begin(), taus->end());
}





// Given a region at iphi/ieta, find the highest region in the surrounding
// regions.
void l1t::Stage1Layer2TauAlgorithmImpPP::findAnnulusInfo(int ieta, int iphi,
    const std::vector<l1t::CaloRegion> & regions,
    double* associatedSecondRegionEt,
    double* associatedThirdRegionEt) const {

  unsigned int neighborsFound = 0;
  double highestNeighborEt = 0;
  double secondNeighborEt = 0;

  for(CaloRegionBxCollection::const_iterator region = regions.begin();
      region != regions.end(); region++) {

    int regionPhi = region->hwPhi();
    int regionEta = region->hwEta();
    unsigned int deltaPhi = iphi - regionPhi;
    if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1) 
      deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi

    unsigned int deltaEta = std::abs(ieta - regionEta);

    if ((deltaPhi + deltaEta) > 0 && deltaPhi < 2 && deltaEta < 2) {

      double regionEt = region->hwPt(); 
      if (regionEt > highestNeighborEt) {
        if(highestNeighborEt!=0) secondNeighborEt=highestNeighborEt;
        highestNeighborEt = regionEt;
      }

      // If we already found all 8 neighbors, we don't need to keep looping
      // over the regions.
      neighborsFound++;
      if (neighborsFound == 8) break;
    }
  }

  // set output
  *associatedSecondRegionEt = highestNeighborEt;
  *associatedThirdRegionEt =secondNeighborEt;
}


