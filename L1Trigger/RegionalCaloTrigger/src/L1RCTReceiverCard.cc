#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTReceiverCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

#include <vector>
using std::vector;

#include <bitset>
using std::bitset;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <string>
using std::string;

L1RCTReceiverCard::L1RCTReceiverCard(int crateNumber,int cardNumber, const L1RCTLookupTables* rctLookupTables) :
  regions(2),crtNo(crateNumber),cardNo(cardNumber),
  rctLookupTables_(rctLookupTables),
  etIn10Bits(2), overFlowBits(2),muonBits(2),tauBits(2)
{
}

L1RCTReceiverCard::~L1RCTReceiverCard(){}

void L1RCTReceiverCard::randomInput(){
  std::vector<unsigned short> input(64);
  for(int i = 0; i<64;i++)
    input.at(i) = rand()&511;
  fillInput(input);
}

void L1RCTReceiverCard::fileInput(char* filename){
  std::vector<unsigned short> input(64);
  unsigned short x;
  std::ifstream instream(filename);
  if(instream){
    for(int i = 0; i<64; i++){
      if(!(instream >> x))
	break;
      else
	input.at(i) = x;
    }
  }
  fillInput(input);
}


//First layer is ecal the second is hcal.
//goes in order of (for crate 0,card 0)
// (Region 1)   (Region 0)
// 28 24 20 16 12 08 04 00
// 29 25 21 17 13 09 05 01
// 30 26 22 18 14 10 06 02
// 31 27 23 19 15 11 07 03 
//
// For card 6 of crate 0 it would look like 
//
// 12 08 04 00
// 13 09 05 01
// 14 10 06 02
// 15 11 07 03
// 16 20 24 28
// 17 21 25 29
// 18 22 26 30
// 19 23 27 31


void L1RCTReceiverCard::fillInput(const std::vector<unsigned short>& input){
  
  std::vector<unsigned short> ecalInput(32);
  std::vector<unsigned short> ecalFG(32);
  std::vector<unsigned short> hcalInput(32);
  std::vector<unsigned short> hcalMuon(32);

  for(int i = 0; i<32; i++){
    ecalInput.at(i) = input.at(i)/2;
    ecalFG.at(i) = input.at(i) & 1;
    hcalInput.at(i) = input.at(i+32)/2;
    hcalMuon.at(i) = input.at(i+32) & 1;
    unsigned long lookup = rctLookupTables_->lookup(ecalInput.at(i),hcalInput.at(i),ecalFG.at(i),crtNo, cardNo, i); // tower number 0-31 now
    unsigned short etIn7Bits = lookup&127;
    unsigned short etIn9Bits = (lookup >> 8)&511;
    unsigned short HE_FGBit = (lookup>>7)&1;
    unsigned short activityBit = (lookup>>17)&1;
    std::vector<unsigned short> indices = towerToRegionMap(i);
    unsigned short r = indices.at(0);
    unsigned short row = indices.at(1);
    unsigned short col = indices.at(2);
    regions.at(r).setEtIn7Bits(row,col,etIn7Bits);
    regions.at(r).setEtIn9Bits(row,col,etIn9Bits);
    regions.at(r).setHE_FGBit(row,col,HE_FGBit);
    regions.at(r).setMuonBit(row,col,hcalMuon.at(i));
    regions.at(r).setActivityBit(row,col,activityBit);
  }

}


vector<unsigned short> L1RCTReceiverCard::towerToRegionMap(int towernum){
  std::vector<unsigned short> returnVec(3);
  unsigned short region;
  unsigned short towerrow;
  unsigned short towercol;
  if(crtNo <9){
    if(cardNo != 6){
      if(towernum < 16){
	region = 0;
	towerrow = towernum%4;
	towercol = 3-(towernum/4);
      }
      else{
	region = 1;
	towerrow = towernum%4;
	towercol = 7-(towernum/4);
      }
    }
    else{
      if(towernum < 16){
	region = 0;
	towerrow = towernum%4;
	towercol = 3-(towernum/4);
      }
      else{
	region = 1;
	towerrow = towernum%4;
	towercol = (towernum/4)-4;
      }
    }
  }
  else{
    if(cardNo != 6){
      if(towernum < 16){
	region = 0;
	towerrow = towernum%4;
	towercol = towernum/4;
      }
      else{
	region = 1;
	towerrow = towernum%4;
	towercol = (towernum/4)-4;
      }
    }
    else{
      if(towernum < 16){
	region = 0;
	towerrow = towernum%4;
	towercol = towernum/4;
      }
      else{
	region = 1;
	towerrow = towernum%4;
	towercol = 7-(towernum/4);
      }
    }
  }
  returnVec.at(0)=region;
  returnVec.at(1)=towerrow;
  returnVec.at(2)=towercol;
  return returnVec;
}    
  


void L1RCTReceiverCard::fillTauBits(){
  for(int i = 0; i<2; i++)
    tauBits.at(i) = calcTauBit(regions.at(i));
}

unsigned short L1RCTReceiverCard::calcTauBit(L1RCTRegion region){
  bitset<4> etaPattern;
  bitset<4> phiPattern;

  bitset<4> badPattern5(string("0101"));
  bitset<4> badPattern7(string("0111"));
  bitset<4> badPattern9(string("1001"));
  bitset<4> badPattern10(string("1010"));
  bitset<4> badPattern11(string("1011"));
  bitset<4> badPattern13(string("1101"));
  bitset<4> badPattern14(string("1110"));
  bitset<4> badPattern15(string("1111"));

  for(int i = 0; i<4; i++){
    phiPattern[i] = region.getActivityBit(i,0) || region.getActivityBit(i,1) ||
      region.getActivityBit(i,2) || region.getActivityBit(i,3);
    etaPattern[i] = region.getActivityBit(0,i) || region.getActivityBit(1,i) ||
      region.getActivityBit(2,i) || region.getActivityBit(3,i);
  }

  bool answer;
  
  if(etaPattern != badPattern5 && etaPattern != badPattern7 && 
     etaPattern != badPattern10 && etaPattern != badPattern11 &&
     etaPattern != badPattern13 && etaPattern != badPattern14 &&
     etaPattern != badPattern15 && phiPattern != badPattern5 && 
     phiPattern != badPattern7 && phiPattern != badPattern10 && 
     phiPattern != badPattern11 && phiPattern != badPattern13 && 
     phiPattern != badPattern14 && phiPattern != badPattern15 &&
     etaPattern != badPattern9 && phiPattern != badPattern9){       // adding in "9"
    //return false;
    answer = false;
  }
  //else return true;
  else {
    answer = true;
  }
  // std::cout << "Tau veto set to " << answer << std::endl;
  return answer;
}

void L1RCTReceiverCard::fillRegionSums(){
  for(int i = 0; i<2; i++){
    etIn10Bits.at(i) = (calcRegionSum(regions.at(i)))/2;
    overFlowBits.at(i) = (calcRegionSum(regions.at(i)) & 1);
  }
}

unsigned short L1RCTReceiverCard::calcRegionSum(L1RCTRegion region){
  unsigned short sum = 0;
  unsigned short overflow = 0;
  for(int i = 0; i<4; i++){
    for(int j = 0; j<4; j++){
      unsigned short towerEt = region.getEtIn9Bits(i,j);
      // If tower is saturated, peg the region to max value
      //if(towerEt == 0x1FF) sum = 0x3FF;  // HARDWARE DOESN'T DO THIS!!
      //else 
      sum = sum + towerEt;
    }
  }
  if(sum > 1023){
    sum = 1023;
    overflow = 1;
  }
  unsigned short sumFullInfo = sum*2 + overflow;
  return sumFullInfo;
}

void L1RCTReceiverCard::fillMuonBits(){
  for(int i = 0; i<2; i++)
    muonBits.at(i) = calcMuonBit(regions.at(i));
}

unsigned short L1RCTReceiverCard::calcMuonBit(L1RCTRegion region){
  unsigned short muonBit = 0;
  for(int i = 0; i<4; i++){
    for(int j = 0; j<4; j++){
      muonBit = muonBit || region.getMuonBit(i,j);
    }
  }
  return muonBit;

}

void L1RCTReceiverCard::print(){
  std::cout <<"Receiver Card " << cardNo << " in Crate " << crtNo <<std::endl;

  for(int i=0;i<2;i++){
    std::cout << "Region " << i << " information" << std::endl;
    regions.at(i).print();
    std::cout << "Region Et sum " << etIn10Bits.at(i) << std::endl;
    std::cout << "Tau Veto Bit " << tauBits.at(i) << std::endl;
    std::cout << "Muon Bit " << muonBits.at(i) << std::endl;
  }
}
