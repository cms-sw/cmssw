///
/// \class l1t::Stage1Layer2EtSumAlgorithmImpPP
///
/// \author: L. Apanasevich
///
/// Description: first iteration of stage 1 jet sums algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EtSumAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

l1t::Stage1Layer2EtSumAlgorithmImpPP::Stage1Layer2EtSumAlgorithmImpPP(const CaloParams* params)
{
  regionETCutForHT=params->regionETCutForHT();
  regionETCutForMET=params->regionETCutForMET();
  minGctEtaForSums=params->minGctEtaForSums();
  maxGctEtaForSums=params->maxGctEtaForSums();

  emScale=params->emScale();
  jetScale=params->jetScale();

  PUSubtract = params->PUSubtract();
  regionSubtraction = params->regionSubtraction();

  //now do what ever initialization is needed
  for(unsigned int i = 0; i < L1CaloRegionDetId::N_PHI; i++) {
    sinPhi.push_back(sin(2. * 3.1415927 * i * 1.0 / L1CaloRegionDetId::N_PHI));
    cosPhi.push_back(cos(2. * 3.1415927 * i * 1.0 / L1CaloRegionDetId::N_PHI));
  }

  // std::cout << "jetScale: " << jetScale  << "\temScale: " << emScale  << std::endl;
}


l1t::Stage1Layer2EtSumAlgorithmImpPP::~Stage1Layer2EtSumAlgorithmImpPP() {


}

double l1t::Stage1Layer2EtSumAlgorithmImpPP::regionPhysicalEt(const l1t::CaloRegion& cand) const {

  return jetScale*cand.hwPt();
}


void l1t::Stage1Layer2EtSumAlgorithmImpPP::processEvent(const std::vector<l1t::CaloRegion> & regions,
							const std::vector<l1t::CaloEmCand> & EMCands,
							      std::vector<l1t::EtSum> * etsums) {

  double sumET = 0;
  double sumEx = 0;
  double sumEy = 0;
  double sumHT = 0;
  double sumHx = 0;
  double sumHy = 0;

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();

  
  //Region Correction will return uncorrected subregions if 
  //PUSubtract is set to False in the config
  RegionCorrection(regions, EMCands, subRegions, regionSubtraction, PUSubtract);

  for(std::vector<CaloRegion>::const_iterator region = subRegions->begin(); region != subRegions->end(); region++) {
    if (region->hwEta() < minGctEtaForSums || region->hwEta() > maxGctEtaForSums) {
      continue;
    }

    double regionET= regionPhysicalEt(*region);

    // if (region->hwPt()>0)
    //   std::cout << "ETA/PHI:" << region->hwEta() <<"/" << region->hwPhi() << "\tPhysical Region ET: " << regionET << "\tHardware Region ET: " << region->hwPt() << std::endl;

    if(regionET >= regionETCutForMET){
      sumET += (int) regionET;
      sumEx += (int) (((double) regionET) * cosPhi[region->hwPhi()]);
      sumEy += (int) (((double) regionET) * sinPhi[region->hwPhi()]);
    }
    if(regionET >= regionETCutForHT) {
      sumHT += (int) regionET;
      sumHx += (int) (((double) regionET) * cosPhi[region->hwPhi()]);
      sumHy += (int) (((double) regionET) * sinPhi[region->hwPhi()]);
    }
  }
  double MET = ((unsigned int) sqrt(sumEx * sumEx + sumEy * sumEy));
  double MHT = ((unsigned int) sqrt(sumHx * sumHx + sumHy * sumHy));

  double physicalPhi = atan2(sumEy, sumEx) + 3.1415927;
  unsigned int iPhiET = L1CaloRegionDetId::N_PHI * physicalPhi / (2 * 3.1415927);

  double physicalPhiHT = atan2(sumHy, sumHx) + 3.1415927;
  unsigned int iPhiHT = L1CaloRegionDetId::N_PHI * (physicalPhiHT) / (2 * 3.1415927);

  // std::cout << "MET:" << MET << "\tHT: " << MHT << std::endl;
  // std::cout << "sumMET:" << sumET << "\tsumHT: " << sumHT << std::endl;

  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *etLorentz =
    new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

  // convert back to hardware ET
  l1t::EtSum etMiss(*etLorentz,EtSum::EtSumType::kMissingEt,MET/jetScale ,0,iPhiET,0);
  l1t::EtSum htMiss(*etLorentz,EtSum::EtSumType::kMissingHt,MHT/jetScale ,0,iPhiHT,0);
  l1t::EtSum etTot (*etLorentz,EtSum::EtSumType::kTotalEt,sumET/jetScale,0,0,0);
  l1t::EtSum htTot (*etLorentz,EtSum::EtSumType::kTotalHt,sumHT/jetScale ,0,0,0);

  etsums->push_back(etMiss);
  etsums->push_back(htMiss);
  etsums->push_back(etTot);
  etsums->push_back(htTot);

}

