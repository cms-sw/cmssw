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
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"


l1t::Stage1Layer2SingleTrackHI::Stage1Layer2SingleTrackHI(CaloParamsHelper* params) : params_(params) {}

l1t::Stage1Layer2SingleTrackHI::~Stage1Layer2SingleTrackHI(){};

void findRegions(const std::vector<l1t::CaloRegion> * sr, std::vector<l1t::Tau> * t, const int etaMask);

void l1t::Stage1Layer2SingleTrackHI::processEvent(const std::vector<l1t::CaloEmCand> & clusters,
						  const std::vector<l1t::CaloRegion> & regions,
						  std::vector<l1t::Tau> * isoTaus,
						  std::vector<l1t::Tau> * taus)
{
  int etaMask = params_->tauRegionMask();

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Tau> *preGtEtaTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *preGtTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *unsortedTaus = new std::vector<l1t::Tau>();


  HICaloRingSubtraction(regions, subRegions, params_);
  findRegions(subRegions, preGtTaus, etaMask);
  TauToGtPtScales(params_, preGtTaus, unsortedTaus);
  SortTaus(unsortedTaus, preGtEtaTaus);
  //SortTaus(preGtTaus, unsortedTaus);
  //TauToGtPtScales(params_, unsortedTaus, preGtEtaTaus);
  TauToGtEtaScales(params_, preGtEtaTaus, taus);

  delete subRegions;
  delete preGtTaus;

  isoTaus->resize(4);
  //taus->resize(4);

  const bool verbose = false;
  const bool hex = true;
  if(verbose)
  {
    if(hex)
    {
      std::cout << "Taus" << std::endl;
      l1t::Tau ataus[8];
      for(std::vector<l1t::Tau>::const_iterator itTau = taus->begin();
	  itTau != taus->end(); ++itTau){
	ataus[itTau - taus->begin()] = *itTau;
      }
      //std::cout << "Taus (hex)" << std::endl;
      std::cout << std::hex << pack16bits(ataus[0].hwPt(), ataus[0].hwEta(), ataus[0].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ataus[1].hwPt(), ataus[1].hwEta(), ataus[1].hwPhi());
      // std::cout << " ";
      // std::cout << std::hex << pack16bits(ataus[4].hwPt(), ataus[4].hwEta(), ataus[4].hwPhi());
      // std::cout << " ";
      // std::cout << std::hex << pack16bits(ataus[5].hwPt(), ataus[5].hwEta(), ataus[5].hwPhi());
      std::cout << std::endl;
      std::cout << std::hex << pack16bits(ataus[2].hwPt(), ataus[2].hwEta(), ataus[2].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ataus[3].hwPt(), ataus[3].hwEta(), ataus[3].hwPhi());
      // std::cout << " ";
      // std::cout << std::hex << pack16bits(ataus[6].hwPt(), ataus[6].hwEta(), ataus[6].hwPhi());
      // std::cout << " ";
      // std::cout << std::hex << pack16bits(ataus[7].hwPt(), ataus[7].hwEta(), ataus[7].hwPhi());
      std::cout << std::endl;
    } else {
      std::cout << "Taus" << std::endl;
      for(std::vector<l1t::Tau>::const_iterator iTau = taus->begin(); iTau != taus->end(); ++iTau)
      {
	unsigned int packed = pack15bits(iTau->hwPt(), iTau->hwEta(), iTau->hwPhi());
	std::cout << bitset<15>(packed).to_string() << std::endl;
      }
      std::cout << "Isolated Taus" << std::endl;
      for(std::vector<l1t::Tau>::const_iterator iTau = isoTaus->begin(); iTau != isoTaus->end(); ++iTau)
      {
	unsigned int packed = pack15bits(iTau->hwPt(), iTau->hwEta(), iTau->hwPhi());
	std::cout << bitset<15>(packed).to_string() << std::endl;
      }
    }
  }
}

void findRegions(const std::vector<l1t::CaloRegion> * sr, std::vector<l1t::Tau> * t, const int etaMask)
{
  for(std::vector<l1t::CaloRegion>::const_iterator region = sr->begin(); region != sr->end(); region++)
  {
    int tauEta = region->hwEta();
    if(tauEta < 4 || tauEta > 17) continue; // taus CANNOT be in the forward region
    if((etaMask & (1<<tauEta))>>tauEta) continue;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > TauLorentz(0,0,0,0);
    l1t::Tau taucand(*&TauLorentz,region->hwPt(),region->hwEta(),region->hwPhi());

    t->push_back(taucand);
  }
}
