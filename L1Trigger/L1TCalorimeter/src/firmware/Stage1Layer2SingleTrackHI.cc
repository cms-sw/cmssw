#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2TauAlgorithmImp.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"

l1t::Stage1Layer2SingleTrackHI::Stage1Layer2SingleTrackHI() {}

l1t::Stage1Layer2SingleTrackHI::~Stage1Layer2SingleTrackHI(){};

void findRegions(const std::vector<l1t::CaloRegion> * sr, std::vector<l1t::Tau> * t);

void l1t::Stage1Layer2SingleTrackHI::processEvent(/*const std::vector<l1t::CaloStage1> & clusters,*/
  const std::vector<l1t::CaloEmCand> & clusters,
  const std::vector<l1t::CaloRegion> & regions,
  const std::vector<l1t::Jet> * jets,
  std::vector<l1t::Tau> * taus)
{
	std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  	HICaloRingSubtraction(regions, subRegions);
        findRegions(subRegions, taus);

	delete subRegions;
}

void findRegions(const std::vector<l1t::CaloRegion> * sr, std::vector<l1t::Tau> * t)
{
  int regionETMax = 0;
  int regionETMaxEta = -1;
  int regionETMaxPhi = -1;

  for(std::vector<l1t::CaloRegion>::const_iterator region = sr->begin(); region != sr->end(); region++)
  {
    int regionET = region->hwPt();
    if (regionET > regionETMax)
    {
      regionETMax = regionET;
      regionETMaxEta = region->hwEta();
      regionETMaxPhi = region->hwPhi();
    }
  }
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *TauLorentz
    = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
  l1t::Tau taucand(*TauLorentz,regionETMax,regionETMaxEta,regionETMaxPhi);

  //don't push a taucand we didn't actually find
  if(taucand.hwPt() > 0)
    t->push_back(taucand);

}
