#ifndef L1RCTReceiverCard_h
#define L1RCTReceiverCard_h

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <bitset>
#include <string>
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRegion.h"

class L1RCTLookupTables;

class L1RCTReceiverCard {

 public:

  L1RCTReceiverCard(int crateNumber, int cardNumber, const L1RCTLookupTables *rctLookupTables);
  ~L1RCTReceiverCard();

  //Information needed to identify cards
  int crateNumber() {return crtNo;}
  int cardNumber() {return cardNo;}

  //Takes in a 64 element vector of unsigned shorts.
  //First layer is ecal the second is hcal.
  //goes in order of (for crate 0,card 0)
  // (Region 1)   (Region 0)
  // 29 25 21 17 13 09 05 01
  // 30 26 22 18 14 10 06 02
  // 31 27 23 19 15 11 07 03
  // 32 28 24 20 16 12 08 04 
  //
  // For card 6 of crate 0 it would look like 
  //
  // 13 09 05 01
  // 14 10 06 02
  // 15 11 07 03
  // 16 12 08 04
  // 17 21 25 29
  // 18 22 26 30
  // 19 23 27 31
  // 20 24 28 32
  //
  //In either case it is set up as so that 0-31 are the 8bit ecal energies
  //plus the fine grain bit, and 32-63 are the 8bit hcal energies plus
  //the muon bit.
  void fillInput(const std::vector<unsigned short>& input);
  void fillTauBits();
  void fillRegionSums();
  void fillMuonBits();
   
  //For each of the following functions the appropriate arguments are
  //0 or 1
  L1RCTRegion *getRegion(int i) {
    return &regions.at(i);
  }
  unsigned short getTauBitRegion(int i) {return tauBits.at(i);}
  unsigned short getMuonBitRegion(int i) {return muonBits.at(i);}
  unsigned short getOverFlowBitRegion(int i) {return overFlowBits.at(i);}
  unsigned short getEtIn10BitsRegion(int i) {return etIn10Bits.at(i);}

  std::vector<unsigned short> towerToRegionMap(int towernum);

  void print();

  void printEdges(){
    regions.at(0).printEdges();
    regions.at(1).printEdges();
  }

  void randomInput();
  void fileInput(char* filename);

 private:
 
  std::vector<L1RCTRegion> regions;
  
  unsigned short calcRegionSum(L1RCTRegion region);
  unsigned short calcTauBit(L1RCTRegion region);
  unsigned short calcMuonBit(L1RCTRegion region);
  unsigned short crtNo;
  unsigned short cardNo;

  const L1RCTLookupTables* rctLookupTables_;

  std::vector<unsigned short> etIn10Bits;
  std::vector<unsigned short> overFlowBits;
  std::vector<unsigned short> muonBits;
  std::vector<unsigned short> tauBits;

  //No default constructor, no copy constructor,
  //and no assignment operator
  L1RCTReceiverCard();
};
#endif
