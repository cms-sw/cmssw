#ifndef L1RCT_h
#define L1RCT_h

#include <vector>
#include <bitset>
#include <iostream>
#include <iomanip>
#include <string>

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCrate.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTNeighborMap.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class L1RCTLookupTables;

class L1RCT {

 public:
  
  L1RCT(const L1RCTLookupTables* rctLookupTables);

  //Organize input for consumption by the cards
  void input();

  //For testing accept external input
  void input(const std::vector<std::vector<std::vector<unsigned short> > >& barrelIn,
	     const std::vector<std::vector<unsigned short> >& hfIn);

  //Should send commands to all crates to send commands to all RCs to
  //process the input data and then send it on to the EICs and then
  //to the JSCs
  void processEvent();

  void fileInput(const char* filename);       // added "const" also in .cc

  void digiInput(const EcalTrigPrimDigiCollection& ecalCollection, 
		 const HcalTrigPrimDigiCollection& hcalCollection);
  
  void randomInput();

  void print();
  void printCrate(int i){
    crates.at(i).print();
  }
  void printJSC(int i){
    crates.at(i).printJSC();
  }
  void printJSC(){
    for(int i=0;i<18;i++){
      std::cout << "JSC for Crate " << i << std::endl;
      crates.at(i).printJSC();
    }
  }
  void printRC(int i, int j){
    crates.at(i).printRC(j);
  }
  void printEIC(int i, int j){
    crates.at(i).printEIC(j);
  }
  void printEICEdges(int i, int j){
    crates.at(i).printEICEdges(j);
  }

  L1CaloEmCollection getIsolatedEGObjects(unsigned crate);

  L1CaloEmCollection getNonisolatedEGObjects(unsigned crate);

  std::vector<unsigned short> getJetRegions(unsigned crate){
    return crates.at(crate).getJetRegions();
  }

  std::vector<L1CaloRegion> getRegions(unsigned crate);

  unsigned short ecalCompressedET(int crate, int card, int tower)
    {return barrel.at(crate).at(card).at(tower) / 2;}
  unsigned short ecalFineGrainBit(int crate, int card, int tower)
    {return barrel.at(crate).at(card).at(tower) & 1;}
  unsigned short hcalCompressedET(int crate, int card, int tower)
    {return barrel.at(crate).at(card).at(tower+32) / 2;}
  unsigned short hcalFineGrainBit(int crate, int card, int tower)
    {return barrel.at(crate).at(card).at(tower+32) & 1;}
  unsigned short hfCompressedET(int crate, int tower)
    {return hf.at(crate).at(tower) / 2;}
  unsigned short hfFineGrainBit(int crate, int tower)
    {return hf.at(crate).at(tower) & 1;}

 private:
  
  L1RCT();  // Do not implement this

  const L1RCTLookupTables* rctLookupTables_;

  //Helper methods called by constructor
  //Set all the neighbors properly.
  //Will make use of the internal neighborMap
  void makeCrates();
  void configureCards();
  void shareNeighbors();

  L1RCTRegion empty;

  //Helper class containing information to set all the neighbors for
  //the receiver cards.  We will use the methods
  //north,south,west,east,se,sw,ne,nw,which each take in the
  //indices of the region in question and then return the indices
  //of the region that is the corresponding neighbor to be used.
  L1RCTNeighborMap neighborMap;

  //Vector of all 18 crates.
  //Will follow numbering convention listed
  //in the CaloTrigger Tower Mapping
  //So 0->8 are eta -5 -> 0
  //While 9-17 are eta 0 -> 5
  //Crate i and crate i+9 are next to each other  
  std::vector<L1RCTCrate> crates;

  //Data for processing is organized into the crates and cards
  //in two multilayered vectors of vectors.
  //The first is of the actual barrel information.
  //18 crates -> 7 RCs -> 64 unsigned shorts per RC
  //so it should be a std::vector<std::vector<std::vector<unsigned short> > >
  //The second is of the HF regions which is just of type
  //std::vector<std::vector<unsigned short> >
  std::vector<std::vector<std::vector<unsigned short> > > barrel;
  std::vector<std::vector<unsigned short> > hf;

};

#endif
