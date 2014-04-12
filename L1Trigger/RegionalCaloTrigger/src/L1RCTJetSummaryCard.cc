#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTJetSummaryCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h" 
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include <iostream>
using std::cout;
using std::endl;
#include <vector>
using std::vector;
#include <bitset>
using std::bitset;

L1RCTJetSummaryCard::L1RCTJetSummaryCard(int crtNo, const L1RCTLookupTables* rctLookupTables) : 
  crtNo(crtNo),
  rctLookupTables_(rctLookupTables),
  isolatedEGObjects(4),
  nonisolatedEGObjects(4),
  jetRegions(22),
  HFRegions(8),
  barrelRegions(14),
  mipBits(0),
  quietBits(0),
  tauBits(0),
  overFlowBits(0),
  hfFineGrainBits(8)
  //quietThreshold(3)
{
}

void L1RCTJetSummaryCard::fillHFRegionSums(const std::vector<unsigned short>& hfRegionSums){
  //std::cout << "JSC.fillHFRegionSums() entered" << std::endl;
  for(int i=0;i<8;i++){
    //std::cout << "filling hf region at " << i << std::endl;
    HFRegions.at(i) = rctLookupTables_->lookup( (hfRegionSums.at(i)/2), crtNo, 999, i);
    //std::cout << "hf region " << i << " et filled" << std::endl;
    hfFineGrainBits.at(i) = (hfRegionSums.at(i)&1);
    //std::cout << "hf region " << i << " fine grain bit filled" << std::endl;
  }
}

void L1RCTJetSummaryCard::fillJetRegions(){
  if(crtNo<9){
    for(int i = 0; i<4;i++){
      jetRegions.at(i) = HFRegions.at(i);
      jetRegions.at(i+11) = HFRegions.at(i+4);
    }
    jetRegions.at(4) = barrelRegions.at(12);
    jetRegions.at(5) = barrelRegions.at(9);
    jetRegions.at(6) = barrelRegions.at(8);
    jetRegions.at(7) = barrelRegions.at(5);
    jetRegions.at(8) = barrelRegions.at(4);
    jetRegions.at(9) = barrelRegions.at(1);
    jetRegions.at(10) = barrelRegions.at(0);

    jetRegions.at(15) = barrelRegions.at(13);
    jetRegions.at(16) = barrelRegions.at(11);
    jetRegions.at(17) = barrelRegions.at(10);
    jetRegions.at(18) = barrelRegions.at(7);
    jetRegions.at(19) = barrelRegions.at(6);
    jetRegions.at(20) = barrelRegions.at(3);
    jetRegions.at(21) = barrelRegions.at(2);
  }
  if(crtNo>=9){
    jetRegions.at(0) = barrelRegions.at(0);
    jetRegions.at(1) = barrelRegions.at(1);
    jetRegions.at(2) = barrelRegions.at(4);
    jetRegions.at(3) = barrelRegions.at(5);
    jetRegions.at(4) = barrelRegions.at(8);
    jetRegions.at(5) = barrelRegions.at(9);
    jetRegions.at(6) = barrelRegions.at(12);
    
    jetRegions.at(11) = barrelRegions.at(2);
    jetRegions.at(12) = barrelRegions.at(3);
    jetRegions.at(13) = barrelRegions.at(6);
    jetRegions.at(14) = barrelRegions.at(7);
    jetRegions.at(15) = barrelRegions.at(10);
    jetRegions.at(16) = barrelRegions.at(11);
    jetRegions.at(17) = barrelRegions.at(13);
    for(int i = 0; i<4;i++){
      jetRegions.at(i+7) = HFRegions.at(i);
      jetRegions.at(i+18) = HFRegions.at(i+4);
    }
  }
}

void L1RCTJetSummaryCard::fillIsolatedEGObjects(const std::vector<unsigned short>& isoElectrons){
  //sort(isoElectrons.begin(),isoElectrons.end());
  //reverse(isoElectrons.begin(),isoElectrons.end());

  std::vector<unsigned short> isoCards03(8);
  std::vector<unsigned short> isoCards46(8);
  std::vector<unsigned short> sortIso(8);

  for (int i = 0; i < 8; i++){
    isoCards03.at(i) = isoElectrons.at(i);
  }
  for (int i = 0; i < 6; i++){
    isoCards46.at(i) = isoElectrons.at(i+8);
  }
  isoCards46.at(6) = 0;
  isoCards46.at(7) = 0;

  asicSort(isoCards03);
  asicSort(isoCards46);

  sortIso.at(0) = isoCards03.at(0);
  sortIso.at(2) = isoCards03.at(2);
  sortIso.at(4) = isoCards03.at(4);
  sortIso.at(6) = isoCards03.at(6);
  sortIso.at(1) = isoCards46.at(0);
  sortIso.at(3) = isoCards46.at(2);
  sortIso.at(5) = isoCards46.at(4);
  sortIso.at(7) = isoCards46.at(6);

  asicSort(sortIso);

  //for(int i = 0; i<4; i++){
    //isolatedEGObjects.at(i) = isoElectrons.at(i);
    //isolatedEGObjects.at(i) = sortIso.at(2*i);
  //}
  isolatedEGObjects.at(0) = sortIso.at(4);
  isolatedEGObjects.at(1) = sortIso.at(6);
  isolatedEGObjects.at(2) = sortIso.at(0);
  isolatedEGObjects.at(3) = sortIso.at(2);
}

void L1RCTJetSummaryCard::fillNonIsolatedEGObjects(const std::vector<unsigned short>& nonIsoElectrons){
  //sort(nonIsoElectrons.begin(),nonIsoElectrons.end());
  //reverse(nonIsoElectrons.begin(),nonIsoElectrons.end());

  std::vector<unsigned short> nonIsoCards03(8);
  std::vector<unsigned short> nonIsoCards46(8);
  std::vector<unsigned short> sortNonIso(8);

  for (int i = 0; i < 8; i++){
    nonIsoCards03.at(i) = nonIsoElectrons.at(i);
  }
  for (int i = 0; i < 6; i++){
    nonIsoCards46.at(i) = nonIsoElectrons.at(i+8);
  }
  nonIsoCards46.at(6) = 0;
  nonIsoCards46.at(7) = 0;

  asicSort(nonIsoCards03);
  asicSort(nonIsoCards46);

  sortNonIso.at(0) = nonIsoCards03.at(0);
  sortNonIso.at(2) = nonIsoCards03.at(2);
  sortNonIso.at(4) = nonIsoCards03.at(4);
  sortNonIso.at(6) = nonIsoCards03.at(6);
  sortNonIso.at(1) = nonIsoCards46.at(0);
  sortNonIso.at(3) = nonIsoCards46.at(2);
  sortNonIso.at(5) = nonIsoCards46.at(4);
  sortNonIso.at(7) = nonIsoCards46.at(6);

  asicSort(sortNonIso);

  //for(int i = 0; i<4; i++){
    //nonisolatedEGObjects.at(i) = nonIsoElectrons.at(i);
    //nonisolatedEGObjects.at(i) = sortNonIso.at(2*i);
  //}
  nonisolatedEGObjects.at(0) = sortNonIso.at(4);
  nonisolatedEGObjects.at(1) = sortNonIso.at(6);
  nonisolatedEGObjects.at(2) = sortNonIso.at(0);
  nonisolatedEGObjects.at(3) = sortNonIso.at(2);
}

void L1RCTJetSummaryCard::fillMIPBits(const std::vector<unsigned short>& mip){
  bitset<14> mips;
  for(int i = 0; i<14; i++)
    mips[i] = mip.at(i);
  mipBits = mips.to_ulong();
}

void L1RCTJetSummaryCard::fillTauBits(const std::vector<unsigned short>& tau){
  bitset<14> taus;
  for(int i = 0; i<14; i++)
    taus[i] = tau.at(i);
  tauBits = taus.to_ulong();
}

void L1RCTJetSummaryCard::fillOverFlowBits(const std::vector<unsigned short>& overflow){
  bitset<14> overflows;
  for(int i = 0; i<14; i++)
    overflows[i] = overflow.at(i);
  overFlowBits = overflows.to_ulong();
}

void L1RCTJetSummaryCard::fillQuietBits(){
  bitset<14> quiet;

  quietThresholdBarrel = rctLookupTables_->rctParameters()->jscQuietThresholdBarrel();
  quietThresholdEndcap = rctLookupTables_->rctParameters()->jscQuietThresholdEndcap();

  // use one threshold for barrel regions (first 8 in list, cards 0-3)
  for(int i = 0; i<8; i++){
    if((barrelRegions.at(i))>quietThresholdBarrel)
      quiet[i] = 0;  //switched 0 and 1
    else
      quiet[i] = 1;
  }
  // use second for endcap regions (last 6 in list, cards 4-6)
  for(int i = 8; i<14; i++){
    if((barrelRegions.at(i))>quietThresholdEndcap)
      quiet[i] = 0;  //switched 0 and 1
    else
      quiet[i] = 1;
  }

  quietBits = quiet.to_ulong();
}

// Sorts the egamma candidates with the algorithm used in the ASIC
void L1RCTJetSummaryCard::asicSort(std::vector<unsigned short>& electrons)
{
  unsigned short temp, temp2;

  asicCompare(electrons);

  // Rotate items prior to next compare

  temp = electrons.at(7);
  electrons.at(7) = electrons.at(5);
  electrons.at(5) = electrons.at(3);
  electrons.at(3) = electrons.at(1);
  electrons.at(1) = temp;

  // Second compare

  asicCompare(electrons);

  // Second rotate, different order this time

  temp = electrons.at(7);
  temp2 = electrons.at(5);
  electrons.at(7) = electrons.at(3);
  electrons.at(5) = electrons.at(1);
  electrons.at(3) = temp;
  electrons.at(1) = temp2;

  // Third compare

  asicCompare(electrons);

  // Third rotate, different again

  temp = electrons.at(1);
  electrons.at(1) = electrons.at(3);
  electrons.at(3) = electrons.at(5);
  electrons.at(5) = electrons.at(7);
  electrons.at(7) = temp;

  // Fourth compare

  asicCompare(electrons);

}

// Used in ASIC sort algorithm
void L1RCTJetSummaryCard::asicCompare(std::vector<unsigned short>& array)
{
  int i;
  unsigned short temp;
  for (i = 0; i < 4; i++)
    {

      unsigned short rank1 = rctLookupTables_->emRank(array.at(2 * i)>>4);
      unsigned short rank2 = rctLookupTables_->emRank(array.at(2 * i + 1)>>4);

      if (rank1 < rank2) // currently bottom 3 bits are rgn,crd
	{
	  temp = array.at(2 * i);
	  array.at(2 * i) = array.at((2 * i) + 1);
	  array.at((2 * i) + 1) = temp;
	}
    }
}


void L1RCTJetSummaryCard::print(){
  std::cout << "tauBits " << tauBits << std::endl;
  std::cout << "MIPBits " << mipBits << std::endl;
  std::cout << "QuietBits " << quietBits << std::endl;
  for(int i=0; i<4;i++) {
    std::cout << "isoElectron " << i << " " << isolatedEGObjects.at(i) << std::endl;;
    std::cout << "nonIsoElectron " << i <<" "<< nonisolatedEGObjects.at(i) << std::endl;
  }
  std::cout << "Jets ";
  for(int i=0; i<22;i++)
    std::cout << jetRegions.at(i) << " ";
  std::cout << std::endl;
}
