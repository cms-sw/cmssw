///
/// \class l1t::Stage1Layer2EtSumAlgorithmImpHW
///
/// \author: Nick Smith (nick.smith@cern.ch)
///
/// Description: hardware emulation of et sum algorithm

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
    cordicPhiValues[i] = static_cast<int>(pow(2.,16)*(((float) i)-36)*M_PI/36);
  }
  for(size_t i=0; i<sines.size(); ++i) {
    sines[i] = static_cast<long>(pow(2,30)*sin(i*20*M_PI/180));
    cosines[i] = static_cast<long>(pow(2,30)*cos(i*20*M_PI/180));
  }
}


l1t::Stage1Layer2EtSumAlgorithmImpHW::~Stage1Layer2EtSumAlgorithmImpHW() {


}

void l1t::Stage1Layer2EtSumAlgorithmImpHW::processEvent(const std::vector<l1t::CaloRegion> & regions,
							const std::vector<l1t::CaloEmCand> & EMCands,
							const std::vector<l1t::Jet> * jets,
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

  // hwPt() is the sum ET+HT in region, for stage 1 this will be
  // the region sum input to MET algorithm
  // In stage 2, we would move to hwEtEm() and hwEtHad() for separate MET/MHT
  // Thresholds will be hardware values not physical
  for (auto& region : *subRegions) {
    if ( region.hwEta() >= etSumEtaMinEt && region.hwEta() <= etSumEtaMaxEt)
    {
      if(region.hwPt() >= etSumEtThresholdEt)
      {
        SimpleRegion r;
        r.ieta = region.hwEta();
        r.iphi = region.hwPhi();
        r.et   = region.hwPt();
        regionEtVect.push_back(r);
      }
    }
    if ( region.hwEta() >= etSumEtaMinHt && region.hwEta() <= etSumEtaMaxHt)
    {
      if(region.hwPt() >= etSumEtThresholdHt)
      {
        SimpleRegion r;
        r.ieta = region.hwEta();
        r.iphi = region.hwPhi();
        r.et   = region.hwPt();
        regionHtVect.push_back(r);
      }
    }
  }

  int sumET, MET, iPhiET;
  std::tie(sumET, MET, iPhiET) = doSumAndMET(regionEtVect, ETSumType::kEmSum);

  int sumHT, MHT, iPhiHT;
  std::tie(sumHT, MHT, iPhiHT) = doSumAndMET(regionHtVect, ETSumType::kHadronicSum);

  //MHT is replaced with MHT/HT
  uint16_t MHToHT=MHToverHT(MHT,sumHT);
  //iPhiHt is replaced by the dPhi between two most energetic jets
  iPhiHT = DiJetPhi(jets);

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
  delete preGtEtSums;

  // Emulator - HDL simulation comparison printout
  const bool verbose = false;
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
l1t::Stage1Layer2EtSumAlgorithmImpHW::doSumAndMET(const std::vector<SimpleRegion>& regionEt, ETSumType sumType)
{
  std::array<int, 18> sumEtaPos{};
  std::array<int, 18> sumEtaNeg{};
  bool inputOverflow(false);
  for (const auto& r : regionEt)
  {
    if ( r.ieta < 11 )
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

// converts phase from 3Q16 to 0-71
// Expects abs(phase) <= 205887 (pi*2^16)
int
l1t::Stage1Layer2EtSumAlgorithmImpHW::cordicToMETPhi(int phase)
{
  assert(abs(phase)<=205887);
  for(size_t i=0; i<cordicPhiValues.size()-1; ++i)
    if ( phase >= cordicPhiValues[i] && phase < cordicPhiValues[i+1] )
      return i;
  // if phase == +205887 (+pi), return zero
  return 0;
}

int l1t::Stage1Layer2EtSumAlgorithmImpHW::DiJetPhi(const std::vector<l1t::Jet> * jets)  const {

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

uint16_t l1t::Stage1Layer2EtSumAlgorithmImpHW::MHToverHT(uint16_t num,uint16_t den)  const {

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
