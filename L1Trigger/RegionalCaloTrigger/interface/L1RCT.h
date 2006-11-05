#ifndef L1RCT_h
#define L1RCT_h

//#include <math>
#include <vector>
#include <bitset>
#include <iostream>
//#include <fstream>
#include <iomanip>
#include <string>

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCrate.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTNeighborMap.h"

//#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "L1Trigger/L1Scales/interface/L1CaloEtScale.h"

class L1RCTLookupTables;

class L1RCT {

 public:
  
  L1RCT(std::string lutFile);

  void setGctEmScale(const L1CaloEtScale* scale);

  //Should organize the input into the crates and cards then pass on
  //the output.  The data will come into the trigger in the form
  //of two multilayered vectors of vectors.
  //The first is of the actual barrel information.
  //18 crates -> 7 RCs -> 64 unsigned shorts per RC
  //so it should be a vector<vector<vector<unsigned short> > >
  //The second is of the HF regions which is just of type
  //vector<vector<unsigned short> >
  void input(vector<vector<vector<unsigned short> > > barrel,
	     vector<vector<unsigned short> > hf);

  //Should send commands to all crates to send commands to all RCs to
  //process the input data and then send it on to the EICs and then
  //to the JSCs
  void processEvent();

  void fileInput(const char* filename);       // added "const" also in .cc

  void digiInput(EcalTrigPrimDigiCollection ecalCollection, HcalTrigPrimDigiCollection hcalCollection);
  void digiTestInput(HcalTrigPrimDigiCollection hcalCollection);

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
      cout << "JSC for Crate " << i << endl;
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

  L1CaloEmCollection getIsolatedEGObjects(int crate);

  L1CaloEmCollection getNonisolatedEGObjects(int crate);

  vector<unsigned short> getJetRegions(int crate){
    return crates.at(crate).getJetRegions();
  }

  vector<L1CaloRegion> getRegions(int crate);

  // Helper methods to convert from trigger tower (iphi, ieta) to RCT (crate, card, tower)
  // Static methods that are globally accessible for now -- this should be designed better!
  static unsigned short calcCrate(unsigned short rct_iphi, short ieta);
  static unsigned short calcCard(unsigned short rct_iphi, unsigned short absIeta);
  static unsigned short calcTower(unsigned short rct_iphi, unsigned short absIeta);
  static short calcIEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower); // negative eta is used
  static unsigned short calcIPhi(unsigned short iCrate, unsigned short iCard, unsigned short iTower);
  
  L1RCTLookupTables* getLUT() {return lut;}

 private:
  
  L1RCT();  // Do not implement this

  //Method called by constructor to set all the neighbors properly.
  //Will make use of the internal neighborMap
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
  vector<L1RCTCrate> crates;

  // scale for converting electron energy to GCT rank
  const L1CaloEtScale* gctEmScale;

  L1RCTLookupTables* lut;

};

#endif
