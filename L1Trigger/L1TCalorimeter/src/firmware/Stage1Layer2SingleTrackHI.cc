// Stage1Layer2SingleTrackHI.cc
// Authors: Michael Northup
//          Alex Barbieri
//
// This is a special-purpose single-track seed trigger which uses the
// Tau channel to communicate with GT. Be wary of any naming scheme
// because we are masquerading as both a tau and track trigger when
// we are really just looking for the hottest region.


#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2TauAlgorithmImp.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

l1t::Stage1Layer2SingleTrackHI::Stage1Layer2SingleTrackHI(CaloParamsStage1* params) : params_(params) {}

l1t::Stage1Layer2SingleTrackHI::~Stage1Layer2SingleTrackHI(){};

void findRegions(const std::vector<l1t::CaloRegion> * sr, std::vector<l1t::Tau> * t);

void l1t::Stage1Layer2SingleTrackHI::processEvent(const std::vector<l1t::CaloEmCand> & clusters,
						  const std::vector<l1t::CaloRegion> & regions,
						  std::vector<l1t::Tau> * isoTaus,
						  std::vector<l1t::Tau> * taus)
{
  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Tau> *preGtEtaTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *preGtTaus = new std::vector<l1t::Tau>();

  HICaloRingSubtraction(regions, subRegions);
  findRegions(subRegions, preGtEtaTaus);
  TauToGtEtaScales(params_, preGtEtaTaus, preGtTaus);
  TauToGtPtScales(params_, preGtTaus, taus);

  delete subRegions;
  delete preGtTaus;
}

void findRegions(const std::vector<l1t::CaloRegion> * sr, std::vector<l1t::Tau> * t)
{
  int regionETMax = 0;
  int regionETMaxEta = -1;
  int regionETMaxPhi = -1;

  for(std::vector<l1t::CaloRegion>::const_iterator region = sr->begin(); region != sr->end(); region++)
  {
    int regionET = region->hwPt();
    if((region->hwEta() < 5) || (region->hwEta() > 16)) continue;
    if (regionET > regionETMax)
    {
      regionETMax = regionET;
      regionETMaxEta = region->hwEta();
      regionETMaxPhi = region->hwPhi();
    }
  }

  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > TauLorentz(0,0,0,0);
  l1t::Tau taucand(*&TauLorentz,regionETMax,regionETMaxEta,regionETMaxPhi);

  //don't push a taucand we didn't actually find
  if(taucand.hwPt() > 0)
    t->push_back(taucand);

}
