// PUSubtractionMethods.cc
// Authors: Alex Barbieri
//          Kalanand Mishra, Fermilab
//          Inga Bucinskaite, UIC
//
// This file should contain the different algorithms used to perform PU, UE subtraction.


//#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"

//#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include <vector>

namespace l1t {

  /// --------------- For heavy ion -------------------------------------
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

      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);

      CaloRegion newSubRegion(*&ldummy, 0, 0, subPt, subEta, subPhi, 0, 0, 0);
      subRegions->push_back(newSubRegion);
    }
  }


  /// --------- New region correction (PUsub, no response correction at the moment) -----------

 void RegionCorrection(const std::vector<l1t::CaloRegion> & regions, 
		       const std::vector<l1t::CaloEmCand> & EMCands, 
		       std::vector<l1t::CaloRegion> *subRegions,
		       std::vector<double> regionPUSParams,
		       std::string regionPUSType) 
{

   int puMult = 0;
   // ------------ This calulates PUM0 -------------------
   for(std::vector<CaloRegion>::const_iterator notCorrectedRegion = regions.begin();
       notCorrectedRegion != regions.end(); notCorrectedRegion++){
      int regionET = notCorrectedRegion->hwPt();
      // cout << "regionET: " << regionET <<endl;
      if (regionET > 0) {puMult++;}
   }

   for(std::vector<CaloRegion>::const_iterator notCorrectedRegion = regions.begin();
       notCorrectedRegion != regions.end(); notCorrectedRegion++){ 
     
     if(regionPUSType == "None") {
       CaloRegion newSubRegion= *notCorrectedRegion;
       subRegions->push_back(newSubRegion);
       continue;
     }


     if (regionPUSType == "PUM0") {
       int regionET = notCorrectedRegion->hwPt();
       int regionEta = notCorrectedRegion->hwEta();
       int regionPhi = notCorrectedRegion->hwPhi();
       
       int regionEtCorr (0);
       // Only non-empty regions are corrected
       if (regionET !=0) {
	 int energyECAL2x1=0;
	 // Find associated 2x1 ECAL energy (EG are calibrated, 
	 // we should not scale them up, it affects the isolation routines)
	 // 2x1 regions have the MAX tower contained in the 4x4 region that its position points to.
	 // This is to not break isolation.
	 for(CaloEmCandBxCollection::const_iterator egCand = EMCands.begin();
	     egCand != EMCands.end(); egCand++) {
	   int et = egCand->hwPt();
	   if(egCand->hwPhi() == regionPhi && egCand->hwEta() == regionEta) {
	     energyECAL2x1=et;
	     break; // I do not really like "breaks"
	   }
	 }
	 
	 //comment out region corrections (below) at the moment, since they're broken
	 
	 //double alpha = regionSF[2*regionEta + 0]; //Region Scale factor (See regionSF_cfi)
	 //double gamma = 2*((regionSF[2*regionEta + 1])/9); //Region Offset. 
	 // It needs to be divided by nine from the jet derived value in the lookup table. 
	 // Multiplied by 2 because gamma is given in regionPhysicalET (=regionEt*regionLSB), 
	 // while we want regionEt= physicalEt/LSB and LSB=.5.
	 
	 
	 //if(!ResponseCorr || regionET<20) {alpha=1;  gamma=0;}
	 double alpha=1;  double gamma=0;
	 
	 
	 int pumbin = (int) puMult/22; //396 Regions. Bins are 22 wide. Dividing by 22 gives which bin# of the 18 bins.
	 
	 double puSub = regionPUSParams[18*regionEta+pumbin]*2;
	 // The values in regionSubtraction are MULTIPLIED by 
	 // RegionLSB=.5 (physicalRegionEt), so to get back unmultiplied 
	 // regionSubtraction we want to multiply the number by 2 
	 // (aka divide by LSB). 
	 
	 int corrpum0pt (0);
	 if(regionET - puSub>0) {
	   int pum0pt = (regionET - puSub-energyECAL2x1); //subtract ECAl energy
	   
	   corrpum0pt = pum0pt*alpha+gamma+energyECAL2x1; 
	   //add back in ECAL energy, calibrate regions(not including the ECAL2x1).
	   if (corrpum0pt <0 || pum0pt<0) {corrpum0pt=0;} //zero floor
	 }
	 regionEtCorr = corrpum0pt;	
       }
       
       ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > lorentz(0,0,0,0);
       CaloRegion newSubRegion(*&lorentz, 0, 0, regionEtCorr, regionEta, regionPhi, 0, 0, 0);
       subRegions->push_back(newSubRegion);
     }
   }
}
  
}
