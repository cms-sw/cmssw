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
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"


using namespace std;
using namespace l1t;


Stage1Layer2TauAlgorithmImpPP::Stage1Layer2TauAlgorithmImpPP(/*const CaloParams & dbPars*/) 
{

}

Stage1Layer2TauAlgorithmImpPP::~Stage1Layer2TauAlgorithmImpPP(){};




void l1t::Stage1Layer2TauAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & EMCands, 
						      const std::vector<l1t::CaloRegion> & regions, 
						      std::vector<l1t::Tau> * taus) {

  tauSeed = 5;
  relativeIsolationCut = 0.1;
  relativeJetIsolationCut = 0.1;

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();

  bool Correct=true;   //  default is to use the corrected regions

  // ----- if using corrected regions ------
  if (Correct) RegionCorrection(regions, EMCands, subRegions);
  else { // --- else just take the uncorrected regions
    for(std::vector<l1t::CaloRegion>::const_iterator region = regions.begin(); 
	region!= regions.end(); region++){
      CaloRegion newSubRegion= *region;
      subRegions->push_back(newSubRegion);
    }
  }


  // ----- need to cluster jets in order to compute jet isolation ----
  std::vector<l1t::Jet> *jets = new std::vector<l1t::Jet>();
  slidingWindowJetFinder(tauSeed, subRegions, jets);



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


    double jetIsolation = JetIsolation(tauEt, region->hwEta(), region->hwPhi(), *jets);


    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *tauLorentz =
      new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::Tau theTau(*tauLorentz, tauEt, region->hwEta(), region->hwPhi());

    if( tauEt >0 && (isolation/tauEt) < relativeIsolationCut 
	&& jetIsolation < relativeJetIsolationCut) 
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





//  Compute jet isolation.
double l1t::Stage1Layer2TauAlgorithmImpPP::JetIsolation(int et, int ieta, int iphi,
							const std::vector<l1t::Jet> & jets) const {

  double isolation = 0;

  for(JetBxCollection::const_iterator jet = jets.begin();
      jet != jets.end(); jet++) {

    if (ieta==jet->hwEta() && iphi==jet->hwPhi())
      isolation = (double) (jet->hwPt() - et);
  }

  // set output
  return isolation/et;
}
