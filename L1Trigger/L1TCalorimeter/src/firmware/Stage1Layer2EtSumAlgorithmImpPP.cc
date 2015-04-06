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
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/JetCalibrationMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"

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


//double l1t::Stage1Layer2EtSumAlgorithmImpPP::regionPhysicalEt(const l1t::CaloRegion& cand) const {
//  return jetLsb*cand.hwPt();
//}

void l1t::Stage1Layer2EtSumAlgorithmImpPP::processEvent(const std::vector<l1t::CaloRegion> & regions,
							const std::vector<l1t::CaloEmCand> & EMCands,
							const std::vector<l1t::Jet> * jets,
							      std::vector<l1t::EtSum> * etsums) {

  unsigned int sumET = 0;
  double sumEx = 0;
  double sumEy = 0;
  unsigned int sumHT = 0;
  double sumHx = 0;
  double sumHy = 0;

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();


  //Region Correction will return uncorrected subregions if
  //regionPUSType is set to None in the config
  double jetLsb=params_->jetLsb();

  int etSumEtaMinEt = params_->etSumEtaMin(0);
  int etSumEtaMaxEt = params_->etSumEtaMax(0);
  //double etSumEtThresholdEt = params_->etSumEtThreshold(0);
  int etSumEtThresholdEt = (int) (params_->etSumEtThreshold(0) / jetLsb);

  int etSumEtaMinHt = params_->etSumEtaMin(1);
  int etSumEtaMaxHt = params_->etSumEtaMax(1);
  //double etSumEtThresholdHt = params_->etSumEtThreshold(1);
  int etSumEtThresholdHt = (int) (params_->etSumEtThreshold(1) / jetLsb);

  std::string regionPUSType = params_->regionPUSType();
  std::vector<double> regionPUSParams = params_->regionPUSParams();
  RegionCorrection(regions, subRegions, regionPUSParams, regionPUSType);

  double towerLsb = params_->towerLsbSum();
  int jetSeedThreshold = floor( params_->jetSeedThreshold()/towerLsb + 0.5);
  // ----- cluster jets for repurposing of MHT phi (use if for angle between leading jet)
  std::vector<l1t::Jet> *unCorrJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> * unSortedJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> * SortedJets = new std::vector<l1t::Jet>();
  slidingWindowJetFinder(jetSeedThreshold, subRegions, unCorrJets);

  //if jetCalibrationType is set to None in the config
  std::string jetCalibrationType = params_->jetCalibrationType();
  std::vector<double> jetCalibrationParams = params_->jetCalibrationParams();
  JetCalibration(unCorrJets, jetCalibrationParams, unSortedJets, jetCalibrationType, towerLsb);

  SortJets(unSortedJets, SortedJets);
  int dijet_phi=DiJetPhi(SortedJets);

  for(std::vector<CaloRegion>::const_iterator region = subRegions->begin(); region != subRegions->end(); region++) {
    if (region->hwEta() < etSumEtaMinEt || region->hwEta() > etSumEtaMaxEt) {
      continue;
    }

    //double regionET= regionPhysicalEt(*region);
    int regionET = region->hwPt();

    if(regionET >= etSumEtThresholdEt){
      sumET += regionET;
      sumEx += (((double) regionET) * cosPhi[region->hwPhi()]);
      sumEy += (((double) regionET) * sinPhi[region->hwPhi()]);
    }
  }

  for(std::vector<CaloRegion>::const_iterator region = subRegions->begin(); region != subRegions->end(); region++) {
    if (region->hwEta() < etSumEtaMinHt || region->hwEta() > etSumEtaMaxHt) {
      continue;
    }

    //double regionET= regionPhysicalEt(*region);
    int regionET = region->hwPt();

    if(regionET >= etSumEtThresholdHt) {
      sumHT += regionET;
      sumHx += (((double) regionET) * cosPhi[region->hwPhi()]);
      sumHy += (((double) regionET) * sinPhi[region->hwPhi()]);
    }
  }

  unsigned int MET = ((unsigned int) sqrt(sumEx * sumEx + sumEy * sumEy));
  unsigned int MHT = ((unsigned int) sqrt(sumHx * sumHx + sumHy * sumHy));

  double physicalPhi = atan2(sumEy, sumEx) + 3.1415927;
  // Global Trigger expects MET phi to be 0-71 (e.g. tower granularity)
  // Although we calculated it with regions, there is some benefit to interpolation.
  unsigned int iPhiET = 4*L1CaloRegionDetId::N_PHI * physicalPhi / (2 * 3.1415927);

  double physicalPhiHT = atan2(sumHy, sumHx) + 3.1415927;
  unsigned int iPhiHT = L1CaloRegionDetId::N_PHI * (physicalPhiHT) / (2 * 3.1415927);

  //std::cout << "MET:" << MET << "\tHT: " << MHT << std::endl;
  //std::cout << "sumMET:" << sumET << "\tsumHT: " << sumHT << std::endl;

  const ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > etLorentz(0,0,0,0);

  // Set quality (i.e. overflow) bits appropriately
  int METqual = 0;
  int MHTqual = 0;
  int ETTqual = 0;
  int HTTqual = 0;
  if(MET >= 0xfff) // MET 12 bits
    METqual = 1;
  if(MHT >= 0x7f)  // MHT 7 bits
    MHTqual = 1;
  if(sumET >= 0xfff)
    ETTqual = 1;
  if(sumHT >= 0xfff)
    HTTqual = 1;



  // scale MHT by sumHT
  // int mtmp = floor (((double) MHT / (double) sumHT)*100 + 0.5);
  // double mtmp = ((double) MHT / (double) sumHT);
  // int rank=params_->HtMissScale().rank(mtmp);
  // MHT=rank;

  uint16_t MHToHT=MHToverHT(MHT,sumHT);
  iPhiHT=dijet_phi;

  l1t::EtSum etMiss(*&etLorentz,EtSum::EtSumType::kMissingEt,MET,0,iPhiET,METqual);
  l1t::EtSum htMiss(*&etLorentz,EtSum::EtSumType::kMissingHt,MHToHT&0x7f,0,iPhiHT,MHTqual);
  l1t::EtSum etTot (*&etLorentz,EtSum::EtSumType::kTotalEt,sumET&0xfff,0,0,ETTqual);
  l1t::EtSum htTot (*&etLorentz,EtSum::EtSumType::kTotalHt,sumHT&0xfff,0,0,HTTqual);

  std::vector<l1t::EtSum> *preGtEtSums = new std::vector<l1t::EtSum>();

  preGtEtSums->push_back(etMiss);
  preGtEtSums->push_back(htMiss);
  preGtEtSums->push_back(etTot);
  preGtEtSums->push_back(htTot);

  EtSumToGtScales(params_, preGtEtSums, etsums);

  delete subRegions;
  delete unCorrJets;
  delete unSortedJets;
  delete SortedJets;
  delete preGtEtSums;

  const bool verbose = false;
  if(verbose)
  {
    for(std::vector<l1t::EtSum>::const_iterator itetsum = etsums->begin();
	itetsum != etsums->end(); ++itetsum){
      // if(EtSum::EtSumType::kMissingEt == itetsum->getType())
      // {
      // 	cout << "Missing Et" << endl;
      // 	cout << bitset<12>(itetsum->hwPt()).to_string() << endl;
      // }
      // if(EtSum::EtSumType::kMissingHt == itetsum->getType())
      // {
      // 	cout << "Missing Ht" << endl;
      // 	cout << bitset<12>(itetsum->hwPt()).to_string() << endl;
      // }
      if(EtSum::EtSumType::kTotalEt == itetsum->getType())
      {
	cout << "Total Et" << endl;
	cout << bitset<12>(itetsum->hwPt()).to_string() << endl;
      }
      if(EtSum::EtSumType::kTotalHt == itetsum->getType())
      {
	cout << "Total Ht" << endl;
	cout << bitset<12>(itetsum->hwPt()).to_string() << endl;
      }
    }
  }
}

int l1t::Stage1Layer2EtSumAlgorithmImpPP::DiJetPhi(const std::vector<l1t::Jet> * jets)  const {

  // cout << "Number of jets: " << jets->size() << endl;

  int dphi = 10; // initialize to negative physical dphi value
  if (jets->size()<2) return dphi; // size() not really reliable as we pad the size to 8 (4cen+4for) in the sorter
  if ((*jets).at(0).hwPt() == 0) return dphi;
  if ((*jets).at(1).hwPt() == 0) return dphi;


  int iphi1 = (*jets).at(0).hwPhi();
  int iphi2 = (*jets).at(1).hwPhi();

  int difference=abs(iphi1-iphi2);

  if ( difference > 9 ) difference= L1CaloRegionDetId::N_PHI - difference ; // make Physical dphi always positive
  return difference;
}

uint16_t l1t::Stage1Layer2EtSumAlgorithmImpPP::MHToverHT(uint16_t num,uint16_t den)  const {

  uint16_t result;
  uint32_t numerator(num),denominator(den);
  
  if(numerator == denominator)
    result = 0x7f;
  else
    {
      numerator = numerator << 7;
      result = numerator/denominator;
      result = result & 0x7f;
    }
  // cout << "Result: " << result << endl;
  
  return result;
}
