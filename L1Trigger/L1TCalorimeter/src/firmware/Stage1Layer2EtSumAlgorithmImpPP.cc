///
/// \class l1t::Stage1Layer2EtSumAlgorithmImpPP
///
/// \author: L. Apanasevich
///
/// Description: first iteration of stage 1 jet sums algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EtSumAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

l1t::Stage1Layer2EtSumAlgorithmImpPP::Stage1Layer2EtSumAlgorithmImpPP(CaloParamsStage1* params) : params_(params)
{
  //now do what ever initialization is needed
  for(unsigned int i = 0; i < L1CaloRegionDetId::N_PHI; i++) {
    sinPhi.push_back(sin(2. * 3.1415927 * i * 1.0 / L1CaloRegionDetId::N_PHI));
    cosPhi.push_back(cos(2. * 3.1415927 * i * 1.0 / L1CaloRegionDetId::N_PHI));
  }
}


l1t::Stage1Layer2EtSumAlgorithmImpPP::~Stage1Layer2EtSumAlgorithmImpPP() {


}

double l1t::Stage1Layer2EtSumAlgorithmImpPP::regionPhysicalEt(const l1t::CaloRegion& cand) const {
 
  return jetLsb*cand.hwPt();
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
  //regionPUSType is set to None in the config
  jetLsb=params_->jetLsb();

  int etSumEtaMinEt = params_->etSumEtaMin(0);
  int etSumEtaMaxEt = params_->etSumEtaMax(0);
  double etSumEtThresholdEt = params_->etSumEtThreshold(0);

  int etSumEtaMinHt = params_->etSumEtaMin(1);
  int etSumEtaMaxHt = params_->etSumEtaMax(1);
  double etSumEtThresholdHt = params_->etSumEtThreshold(1);

  std::string regionPUSType = params_->regionPUSType();
  std::vector<double> regionPUSParams = params_->regionPUSParams();
  RegionCorrection(regions, EMCands, subRegions, regionPUSParams, regionPUSType);

  for(std::vector<CaloRegion>::const_iterator region = subRegions->begin(); region != subRegions->end(); region++) {
    if (region->hwEta() < etSumEtaMinEt || region->hwEta() > etSumEtaMaxEt) {
      continue;
    }

    double regionET= regionPhysicalEt(*region);

    if(regionET >= etSumEtThresholdEt){
      sumET += (int) regionET;
      sumEx += (int) (((double) regionET) * cosPhi[region->hwPhi()]);
      sumEy += (int) (((double) regionET) * sinPhi[region->hwPhi()]);
    }
  }

  for(std::vector<CaloRegion>::const_iterator region = subRegions->begin(); region != subRegions->end(); region++) {
    if (region->hwEta() < etSumEtaMinHt || region->hwEta() > etSumEtaMaxHt) {
      continue;
    }

    double regionET= regionPhysicalEt(*region);

    if(regionET >= etSumEtThresholdHt) {
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

  //std::cout << "MET:" << MET << "\tHT: " << MHT << std::endl;
  //std::cout << "sumMET:" << sumET << "\tsumHT: " << sumHT << std::endl;

  const ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > etLorentz(0,0,0,0);

  // convert back to hardware ET
  l1t::EtSum etMiss(*&etLorentz,EtSum::EtSumType::kMissingEt,MET/jetLsb ,0,iPhiET,0);
  l1t::EtSum htMiss(*&etLorentz,EtSum::EtSumType::kMissingHt,MHT/jetLsb ,0,iPhiHT,0);
  l1t::EtSum etTot (*&etLorentz,EtSum::EtSumType::kTotalEt,sumET/jetLsb,0,0,0);
  l1t::EtSum htTot (*&etLorentz,EtSum::EtSumType::kTotalHt,sumHT/jetLsb ,0,0,0);

  std::vector<l1t::EtSum> *preGtEtSums = new std::vector<l1t::EtSum>();

  preGtEtSums->push_back(etMiss);
  preGtEtSums->push_back(htMiss);
  preGtEtSums->push_back(etTot);
  preGtEtSums->push_back(htTot);

  EtSumToGtScales(params_, preGtEtSums, etsums);

  delete subRegions;
  delete preGtEtSums;

}
