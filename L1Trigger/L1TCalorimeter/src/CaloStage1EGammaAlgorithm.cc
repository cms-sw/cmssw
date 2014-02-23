///step03
/// \class l1t::CaloStage1EGammaAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Kalanand Mishra - Fermilab
///

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1EGammaAlgorithm.h"


void l1t::CaloStage1EGammaAlgorithm::processEvent(const std::vector<l1t::CaloStage1Cluster> & clusters, const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::EGamma> & egammas, std::list<L1GObject> & rlxEGList, std::list<L1GObject> & isoEGList) { 

  egtSeed = 0;
  puLevel = 0.0;
  rlxEGList.clear();
  isoEGList.clear();


  for(std::vector<l1t::EGamma>::const_iterator egtCand =
	egammas.begin();
      egtCand != egammas.end(); egtCand++){
    double et = egtCand->et();
    if(et > egtSeed) {
      for(CaloRegionBxCollection::const_iterator region = regions.begin();
	  region != regions.end(); region++) {
	if(egtCand->phi() == region->phi() &&
	   egtCand->eta() == region->eta())
	  {
            double regionEt = region->et();
            // Debugging
            if (false && egtCand->et() > regionEt) {
              std::cout << "Mismatch!" << std::endl;
              std::cout << "egPhi = " << egtCand->phi() << std::endl;
              std::cout << "egEta = " << egtCand->eta() << std::endl;
              std::cout << "egEt = " << egtCand->et()   << std::endl;
              std::cout << "regionEt = " << regionEt    << std::endl;
              std::cout << "ratio = " << et/regionEt    << std::endl;
            }


	    rlxEGList.push_back(L1GObject(et, egtCand->eta(), 
					  egtCand->phi(), "EG"));
	    rlxEGList.back().associatedRegionEt_ = regionEt;



	    double isolation = regionEt - puLevel - et;   // Core isolation (could go less than zero)
	    double relativeIsolation = isolation / et;
	    
	    
	    if(relativeIsolation < relativeIsolationCut) {
	      // Relative isolation makes it to IsoEG
	      isoEGList.push_back(L1GObject(et, egtCand->eta(), 
					    egtCand->phi(), "IsoEG"));
	    }
	    break;
	  }
      }
    }
  }
  rlxEGList.sort();
  isoEGList.sort();
  rlxEGList.reverse();
  isoEGList.reverse();
}

