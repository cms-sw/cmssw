///
/// \class l1t::Stage1Layer2EtSumAlgorithmImpHW
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
#include <cassert>

l1t::Stage1Layer2EtSumAlgorithmImpHW::Stage1Layer2EtSumAlgorithmImpHW(CaloParamsStage1* params) : params_(params)
{
  //now do what ever initialization is needed
  for(size_t i=0; i<cordicPhiValues.size(); ++i) {
    cordicPhiValues[i] = static_cast<int>(pow(2.,16)*(i-36)*M_PI/36);
  }
  for(size_t i=0; i<sines.size(); ++i) {
    sines[i] = static_cast<long>(pow(2,30)*sin(i*20*M_PI/180));
    cosines[i] = static_cast<long>(pow(2,30)*cos(i*20*M_PI/180));
  }
}


l1t::Stage1Layer2EtSumAlgorithmImpHW::~Stage1Layer2EtSumAlgorithmImpHW() {


}


//double l1t::Stage1Layer2EtSumAlgorithmImpHW::regionPhysicalEt(const l1t::CaloRegion& cand) const {
//  return jetLsb*cand.hwPt();
//}

void l1t::Stage1Layer2EtSumAlgorithmImpHW::processEvent(const std::vector<l1t::CaloRegion> & regions,
							const std::vector<l1t::CaloEmCand> & EMCands,
							      std::vector<l1t::EtSum> * etsums) {

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

  std::vector<SimpleRegion> regionEtVect;
  std::vector<SimpleRegion> regionHtVect;

  for (auto& region : *subRegions) {
    if ( region.hwEta() >= etSumEtaMinEt && region.hwEta() <= etSumEtaMaxEt)
    {
      if(region.etEm() >= etSumEtThresholdEt)
      {
        SimpleRegion r;
        r.ieta = region.hwEta();
        r.iphi = region.hwPhi();
        r.et   = region.hwEtEm();
        regionEtVect.push_back(r);
      }
    }
    if ( region.hwEta() >= etSumEtaMinHt && region.hwEta() <= etSumEtaMaxHt)
    {
      if(region.etHad() >= etSumEtThresholdHt)
      {
        SimpleRegion r;
        r.ieta = region.hwEta();
        r.iphi = region.hwPhi();
        r.et   = region.hwEtHad();
        regionHtVect.push_back(r);
      }
    }
  }

  int sumET, MET, iPhiET;
  std::tie(sumET, MET, iPhiET) = doSumAndMET(regionEtVect, ETSumType::kEmSum);

  int sumHT, MHT, iPhiHT;
  std::tie(sumHT, MHT, iPhiHT) = doSumAndMET(regionHtVect, ETSumType::kHadronicSum);

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

  const ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > etLorentz(0,0,0,0);
  l1t::EtSum etMiss(*&etLorentz,EtSum::EtSumType::kMissingEt,MET&0xfff,0,iPhiET,METqual);
  l1t::EtSum htMiss(*&etLorentz,EtSum::EtSumType::kMissingHt,MHT&0x7f,0,iPhiHT,MHTqual);
  l1t::EtSum etTot (*&etLorentz,EtSum::EtSumType::kTotalEt,sumET&0xfff,0,0,ETTqual);
  l1t::EtSum htTot (*&etLorentz,EtSum::EtSumType::kTotalHt,sumHT&0xfff,0,0,HTTqual);

  std::vector<l1t::EtSum> *preGtEtSums = new std::vector<l1t::EtSum>();

  preGtEtSums->push_back(etMiss);
  preGtEtSums->push_back(htMiss);
  preGtEtSums->push_back(etTot);
  preGtEtSums->push_back(htTot);

  EtSumToGtScales(params_, preGtEtSums, etsums);

  delete subRegions;
  // delete unCorrJets;
  // delete unSortedJets;
  // delete SortedJets;
  delete preGtEtSums;

  const bool verbose = true;
  if(verbose)
  {
    for(std::vector<l1t::EtSum>::const_iterator itetsum = etsums->begin();
	itetsum != etsums->end(); ++itetsum){
      if(EtSum::EtSumType::kMissingEt == itetsum->getType())
      {
      	cout << "Missing Et" << endl;
      	cout << bitset<7>(itetsum->hwPhi()).to_string() << bitset<1>(itetsum->hwQual()).to_string() << bitset<12>(itetsum->hwPt()).to_string() << endl;
      }
      if(EtSum::EtSumType::kMissingHt == itetsum->getType())
      {
      	cout << "Missing Ht" << endl;
      	cout << bitset<1>(itetsum->hwQual()).to_string() << bitset<7>(itetsum->hwPt()).to_string() << bitset<5>(itetsum->hwPhi()).to_string() << endl;
      }
      if(EtSum::EtSumType::kTotalEt == itetsum->getType())
      {
	cout << "Total Et" << endl;
	cout << bitset<1>(itetsum->hwQual()).to_string() << bitset<12>(itetsum->hwPt()).to_string() << endl;
      }
      if(EtSum::EtSumType::kTotalHt == itetsum->getType())
      {
	cout << "Total Ht" << endl;
	cout << bitset<1>(itetsum->hwQual()).to_string() << bitset<12>(itetsum->hwPt()).to_string() << endl;
      }
    }
  }
}

std::tuple<int, int, int>
l1t::Stage1Layer2EtSumAlgorithmImpHW::doSumAndMET(std::vector<SimpleRegion>& regionEt, ETSumType sumType)
{
// if any region et/ht has overflow bit, set sumET overflow bit
// met/mht same, breakout to function
// threshold 15 mht 
// sum phi for -eta for region 0,2,4,6.0 : 18 -eta numbers 15 bit
// sum phi for +eta for region 1,3,5,6.1 : 18 +eta numbers 15 bit
// sum -eta and +eta 15 bit
// sum all for sumET : 19 bit (22*18 fits in 9 bits, started with 10 bits)
// sumET mask to 12 bits both
// if larger set overflow, ovf is thirteenth bit in sumET/HT
// sum phi[x] and -phi[x+9] for x=0..8 : 16 bit 2'sC
// e.g 0-180, 20-200, .. also 280-100, etc. 4 numbers
// treat cos(+-60) special 1/2 e.g. shift 29 bits
// 30 bit sin/cos : 46 bit result still 2'sC
// add x and y values in 49 bit 2'sC
// shift (truncate) to 24 bit cordic input
// CORDIC
// MET get first 12 bits
// if >2^12, set ovf (13th bit)
// MHT get first 7
// if >2^7 set ovf (8th bit)
// phase 3Q16 to 72 (5719)
// by starting at -pi and going CCW around circle try to see if angle <= x < angle+1
// mht shift down truncate to 5
// So etsum from 180 to 185 degress is MET phi of 0
// 13 13 13 8 7 5
  std::array<int, 18> sumEtaPos{};
  std::array<int, 18> sumEtaNeg{};
  bool inputOverflow(false);
  for (const auto& r : regionEt)
  {
    if ( r.ieta < 0 )
      sumEtaNeg[r.iphi] += r.et;
    else
      sumEtaPos[r.iphi] += r.et;
    
    if ( r.et >= (1<<10) )
      inputOverflow = true;
  }

  std::array<int, 18> sumEta{};
  int sumEt(0);
  for(size_t i=0; i<sumEta.size(); ++i)
  {
    assert(sumEtaPos[i] >= 0 && sumEtaNeg[i] >= 0);
    sumEta[i] = sumEtaPos[i] + sumEtaNeg[i];
    sumEt += sumEta[i];
  }
  sumEt = (sumEt % (1<<12)) | ((sumEt >= (1<<12) || inputOverflow) ? (1<<12):0);
  assert(sumEt>=0 && sumEt < (1<<13));

  // 0, 20, 40, 60, 80 degrees
  std::array<int, 5> sumsForCos{};
  std::array<int, 5> sumsForSin{};
  for(size_t iphi=0; iphi<sumEta.size(); ++iphi)
  {
    if ( iphi < 5 )
    {
      sumsForCos[iphi] += sumEta[iphi];
      sumsForSin[iphi] += sumEta[iphi];
    }
    else if ( iphi < 9 )
    {
      sumsForCos[9-iphi] -= sumEta[iphi];
      sumsForSin[9-iphi] += sumEta[iphi];
    }
    else if ( iphi < 14 )
    {
      sumsForCos[iphi-9] -= sumEta[iphi];
      sumsForSin[iphi-9] -= sumEta[iphi];
    }
    else
    {
      sumsForCos[18-iphi] += sumEta[iphi];
      sumsForSin[18-iphi] -= sumEta[iphi];
    }
  }

  long sumX(0l);
  long sumY(0l);
  for(int i=0; i<5; ++i)
  {
    sumX += sumsForCos[i]*cosines[i];
    sumY += sumsForSin[i]*sines[i];
  }
  assert(abs(sumX)<(1l<<48) && abs(sumY)<(1l<<48));
  int cordicX = sumX>>25;
  int cordicY = sumY>>25;

  uint32_t cordicMag(0);
  int cordicPhase(0);
  cordic(cordicX, cordicY, cordicPhase, cordicMag);

  int met(0);
  int metPhi(0);
  if ( sumType == ETSumType::kHadronicSum )
  {
    met  = (cordicMag % (1<<7)) | ((cordicMag >= (1<<7)) ? (1<<7):0);
    metPhi = cordicToMETPhi(cordicPhase) >> 2;
    assert(metPhi >=0 && metPhi < 18);
  }
  else
  {
    met  = (cordicMag % (1<<12)) | ((cordicMag >= (1<<12)) ? (1<<12):0);
    metPhi = cordicToMETPhi(cordicPhase);
    assert(metPhi >=0 && metPhi < 72);
  }

  return std::make_tuple(sumEt, met, metPhi);
}

// phase 3Q16 to 72 (5719)
int
l1t::Stage1Layer2EtSumAlgorithmImpHW::cordicToMETPhi(int phase)
{
  for(size_t i=0; i<cordicPhiValues.size()-1; ++i)
    if ( phase >= cordicPhiValues[i] && phase < cordicPhiValues[i+1] )
      return i;
  return -1;
}
