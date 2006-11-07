#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTJetSummaryCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include <iostream>
using std::cout;
using std::endl;

L1RCTJetSummaryCard::L1RCTJetSummaryCard(int crtNo):isolatedEGObjects(4),
						    nonisolatedEGObjects(4),
						    jetRegions(22),
						    HFRegions(8),
						    barrelRegions(14),
						    mipBits(0),
						    quietBits(0),
						    tauBits(0),
						    overFlowBits(0),
						    hfFineGrainBits(8),
						    crtNo(crtNo),
						    quietThreshold(3)
{
}

void L1RCTJetSummaryCard::fillHFRegionSums(vector<unsigned short> hfRegionSums, L1RCTLookupTables *lut){
  //cout << "JSC.fillHFRegionSums() entered" << endl;
  for(int i=0;i<8;i++){
    //cout << "filling hf region at " << i << endl;
    HFRegions.at(i) = lut->lookup( (hfRegionSums.at(i)/2), crtNo, 999, i);
    //cout << "hf region " << i << " et filled" << endl;
    hfFineGrainBits.at(i) = (hfRegionSums.at(i)&1);
    //cout << "hf region " << i << " fine grain bit filled" << endl;
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

void L1RCTJetSummaryCard::fillIsolatedEGObjects(vector<unsigned short> isoElectrons){
  sort(isoElectrons.begin(),isoElectrons.end());
  reverse(isoElectrons.begin(),isoElectrons.end());
  for(int i = 0; i<4; i++)
    isolatedEGObjects.at(i) = isoElectrons.at(i);
}

void L1RCTJetSummaryCard::fillNonIsolatedEGObjects(vector<unsigned short> nonIsoElectrons){
 sort(nonIsoElectrons.begin(),nonIsoElectrons.end());
 reverse(nonIsoElectrons.begin(),nonIsoElectrons.end());
 for(int i = 0; i<4; i++)
    nonisolatedEGObjects.at(i) = nonIsoElectrons.at(i);
}

void L1RCTJetSummaryCard::fillMIPBits(vector<unsigned short> mip){
  bitset<14> mips;
  for(int i = 0; i<14; i++)
    mips[i] = mip.at(i);
  mipBits = mips.to_ulong();
}

void L1RCTJetSummaryCard::fillTauBits(vector<unsigned short> tau){
  bitset<14> taus;
  for(int i = 0; i<14; i++)
    taus[i] = tau.at(i);
  tauBits = taus.to_ulong();
}

void L1RCTJetSummaryCard::fillOverFlowBits(vector<unsigned short> overflow){
  bitset<14> overflows;
  for(int i = 0; i<14; i++)
    overflows[i] = overflow.at(i);
  overFlowBits = overflows.to_ulong();
}

void L1RCTJetSummaryCard::fillQuietBits(){
  bitset<14> quiet;
  for(int i = 0; i<14; i++){
    if((barrelRegions.at(i))>quietThreshold)
      quiet[i] = 1;
    else
      quiet[i] = 0;
  }

  quietBits = quiet.to_ulong();
}

void L1RCTJetSummaryCard::print(){
  cout << "tauBits " << tauBits << endl;
  cout << "MIPBits " << mipBits << endl;
  cout << "QuietBits " << quietBits << endl;
  for(int i=0; i<4;i++) {
    cout << "isoElectron " << i << " " << isolatedEGObjects.at(i) << endl;;
    cout << "nonIsoElectron " << i <<" "<< nonisolatedEGObjects.at(i) << endl;
  }
  cout << "Jets ";
  for(int i=0; i<22;i++)
    cout << jetRegions.at(i) << " ";
  cout << endl;
}
