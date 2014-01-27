#include "L1Trigger/L1TCalorimeter/interface/CaloStage1TauAlgorithmImp.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"

using namespace std;
using namespace l1t;

CaloStage1SingleTrackHI::CaloStage1SingleTrackHI() {}

CaloStage1SingleTrackHI::~CaloStage1SingleTrackHI(){};

void findRegions(const std::vector<l1t::CaloRegion> * sr, std::vector<l1t::Tau> * t);

void CaloStage1SingleTrackHI::processEvent(/*const std::vector<l1t::CaloStage1Cluster> & clusters,*/
  const std::vector<l1t::CaloEmCand> & clusters,
  const std::vector<l1t::CaloRegion> & regions,
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
  int regionETMaxEta = 0;
  int regionETMaxPhi = 0;

  for(std::vector<CaloRegion>::const_iterator region = sr->begin(); region != sr->end(); region++)
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

  t->push_back(taucand);

}
