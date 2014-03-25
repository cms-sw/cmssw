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
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"

void l1t::Stage1Layer2TauAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & EMCands, 
						      const std::vector<l1t::CaloRegion> & regions, 
						      std::vector<l1t::Tau> * taus) {

  tauSeed = 0;
  relativeIsolationCut = 0.2;

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  RegionCorrection(regions, EMCands, subRegions);



  for(CaloRegionBxCollection::const_iterator region = subRegions->begin();
      region != subRegions->end(); region++) {

    int regionEt = region->hwPt(); 
    if(regionEt < tauSeed) continue;
            
    double isolation;
    int associatedSecondRegionEt = 
      AssociatedSecondRegionEt(region->hwEta(), region->hwPhi(),
			       *subRegions, isolation);
    
    int tauEt=regionEt;
    if(associatedSecondRegionEt>tauSeed) tauEt +=associatedSecondRegionEt;


    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *tauLorentz =
      new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::Tau theTau(*tauLorentz, tauEt, region->hwEta(), region->phi());

    if( (isolation / tauEt -1.0) < relativeIsolationCut) 
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
// regions. Also compute isolation.

int l1t::Stage1Layer2TauAlgorithmImpPP::AssociatedSecondRegionEt(int ieta, int iphi,
								 const std::vector<l1t::CaloRegion> & regions, 
								 double& isolation) const {
  int highestNeighborEt = 0;
  isolation = 0;

  for(CaloRegionBxCollection::const_iterator region = regions.begin();
      region != regions.end(); region++) {

    int regionPhi = region->hwPhi();
    int regionEta = region->hwEta();
    unsigned int deltaPhi = iphi - regionPhi;
    if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1) 
      deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi

    unsigned int deltaEta = std::abs(ieta - regionEta);

    if ((deltaPhi + deltaEta) > 0 && deltaPhi < 2 && deltaEta < 2) {

      int regionEt = region->hwPt(); 
      isolation += regionEt;
      if (regionEt > highestNeighborEt) highestNeighborEt = regionEt;
    }
  }

  // set output
  return highestNeighborEt;
}


