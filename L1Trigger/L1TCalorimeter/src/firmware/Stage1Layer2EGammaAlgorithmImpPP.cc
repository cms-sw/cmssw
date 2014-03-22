///step03
/// \class l1t::Stage1Layer2EGammaAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Kalanand Mishra - Fermilab
///

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EGammaAlgorithmImp.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"


void l1t::Stage1Layer2EGammaAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & clusters, const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::EGamma> & egammas) {

  egtSeed = 0;
  puLevel = 0.0;
  relativeIsolationCut = 0.2;

  for(CaloEmCandBxCollection::const_iterator egCand = clusters.begin();
	  egCand != clusters.end(); egCand++) {

     double eg_et = egCand->hwPt();
     int eg_eta = egCand->hwEta();
     int eg_phi = egCand->hwPhi();
     if(eg_et < egtSeed) continue;


     /// -----  Compute isolation sum --------------------  
     double isolation = 0.0;
     for(CaloRegionBxCollection::const_iterator region = regions.begin();
         region != regions.end(); region++) {

        int regionPhi = region->hwPhi();
        int regionEta = region->hwEta();
        unsigned int deltaPhi = eg_phi - regionPhi;        

        if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1) 
           deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi

        unsigned int deltaEta = std::abs(eg_eta - regionEta);
        if ((deltaPhi + deltaEta) > 0 && deltaPhi < 2 && deltaEta < 2) 
           isolation += region->hwPt(); //regionPhysicalEt(*region);
     }

     isolation -=  puLevel;   // Core isolation (could go less than zero)

 
     ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *egLorentz =
        new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

     int quality = 0;
     int isoFlag = 0;
     if(isolation / eg_et < relativeIsolationCut) isoFlag  = 1;

     l1t::EGamma theEG(*egLorentz, eg_et, eg_eta, eg_phi, quality, isoFlag);
      egammas.push_back(theEG);
  }


   //the EG candidates should be sorted, highest pT first.
   // do not truncate the EG list, GT converter handles that
   auto comp = [&](l1t::EGamma i, l1t::EGamma j)-> bool {
        return (i.hwPt() < j.hwPt() ); 
    };

   std::sort(egammas.begin(), egammas.end(), comp);
   std::reverse(egammas.begin(), egammas.end());
}
