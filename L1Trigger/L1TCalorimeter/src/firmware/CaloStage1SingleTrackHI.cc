#include "L1Trigger/L1TCalorimeter/interface/CaloStage1SingleTrackHI.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

using namespace std;
using namespace l1t;

CaloStage1SingleTrackHI::CaloStage1SingleTrackHI() : regionLSB_(0.5) {}

CaloStage1SingleTrackHI::~CaloStage1SingleTrackHI(){};

void puSubtraction(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> subRegions);
void findRegions(const std::vector<l1t::CaloRegion> & sr, std::vector<l1t::Tau> & t);
double regionPhysicalEt(const l1t::CaloRegion& cand);

void CaloStage1SingleTrackHI::processEvent(/*const std::vector<l1t::CaloStage1Cluster> & clusters,*/
			      const std::vector<l1t::CaloEmCand> & clusters,	
                              const std::vector<l1t::CaloRegion> & regions,
                              std::vector<l1t::Tau> & taus)
{
	std::vector<l1t::CaloRegion> subRegions;
  	puSubtraction(regions, subRegions);
        findRegions(subRegions, taus);
};

void puSubtraction(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> subRegions)
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

  for(vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++){
    r_puLevelHI[region->hwEta()] += region->hwPt();
    etaCount[region->hwEta()]++;
  }

  for(unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i)
  {
    puLevelHI[i] = floor(r_puLevelHI[i]/etaCount[i] + 0.5);
  }

  for(vector<CaloRegion>::const_iterator region = regions.begin(); region!= regions.end(); region++){
    int subPt = std::max(0, region->hwPt() - puLevelHI[region->hwEta()]);
    int subEta = region->hwEta();
    int subPhi = region->hwPhi();

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *lorentz =
      new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    CaloRegion newSubRegion(*lorentz, 0, 0, subPt, subEta, subPhi, 0, 0, 0);
    subRegions.push_back(newSubRegion);
  }
};

void findRegions(const std::vector<l1t::CaloRegion> & sr, std::vector<l1t::Tau> & t)
{
	double regionETMax = 0.0;
	double regionETMaxEta = 0.0;
	double regionETMaxPhi = 0.0;

	for(std::vector<CaloRegion>::const_iterator region = sr.begin(); region != sr.end(); region++)
	{
		double regionET = regionPhysicalEt(*region);
		if (regionET > regionETMax)
		{
			regionETMax = regionET;
			//regionETMaxEta = *region.hwEta();
			//regionETMaxPhi = *region.hwPhi();
		}
			
	}
	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *TauLorentz = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
	l1t::Tau taucand(*TauLorentz,regionETMax,regionETMaxEta,regionETMaxPhi);

	t.push_back(taucand);
		
};

double regionPhysicalEt(const l1t::CaloRegion& cand) 
	{
	  //return regionLSB_*cand.hwPt();
	  return 0.0;
   	};


