#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

#include <vector>
using std::vector;

#include <fstream>
#include <string>

#include <iostream>
using std::ostream;
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>

//#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

//Main method to process a single event, hence the name.
//First it sets up all the neighbors, sharing the pointers to the proper
//regions.  This is done via the neighborMap auxiliary class, which
//is very dry and contains the proper mappings from crate,card,and
//region numbers to the crate,card, and region numbers of the neighbors.
//The next step is to pass along the pointers for the regions
//to their corresponding Electron Isolation Card
//This is done in the crate method fillElectronIsolationCards
//Then the actual processing of the data begins with the
//processReceiverCards and processElectronIsolationCards methods.
//Next the region sums, tau bits, mip bits, and electron
//candidates are passed onto the Jet Summary Card, and that's where
//the data flow ends for the Regional CaloTrigger.
void L1RCT::processEvent(){
  for(int i=0; i<18;i++)
    crates.at(i).processReceiverCards();
  shareNeighbors();  
  for(int i=0; i<18;i++){
    crates.at(i).fillElectronIsolationCards();
    crates.at(i).processElectronIsolationCards();
    crates.at(i).fillJetSummaryCard();
    crates.at(i).processJetSummaryCard();
  }
}

void L1RCT::makeCrates()
{
  for(int i = 0; i<18; i++){
    L1RCTCrate c(i, rctLookupTables_);
    crates.push_back(c);
  }
}

L1RCT::L1RCT(const L1RCTLookupTables* rctLookupTables) : 
  rctLookupTables_(rctLookupTables),
  empty(),
  neighborMap(),
  barrel(18,std::vector<std::vector<unsigned short> >(7,std::vector<unsigned short>(64))),
  hf(18,std::vector<unsigned short>(8))
{
  makeCrates();
}

void L1RCT::input()
{
  for(int i = 0; i<18; i++){
    crates.at(i).input(barrel.at(i),hf.at(i));
  }
}

void L1RCT::input(const std::vector<std::vector<std::vector<unsigned short> > >& barrelIn,
		  const std::vector<std::vector<unsigned short> >& hfIn)
{
  for(int i = 0; i<18; i++){
    crates.at(i).input(barrelIn.at(i),hfIn.at(i));
  }
}

//This is a method for taking input from a file.  Any entries in excess
//of 18*7*64 will simply be ignored.  This *only* fills input for a single
//event.  At the moment you cannot put a ton of data and have it be
//read as separate events.
void L1RCT::fileInput(const char* filename){            // added "const" also in .h
  unsigned short x;
  std::ifstream instream(filename);
  if(instream){
    for(int i = 0; i<18;i++){
      for(int j = 0; j<7; j++){
	for(int k = 0; k<64; k++){
	  if(instream >> x){
	    unsigned short bit = x/256;             // added J.Leonard Aug. 16 06
	    unsigned short energy = x&255;          //
	    unsigned short input = energy*2 + bit;  //
	    barrel.at(i).at(j).at(k) = input;
	  }
	  else
	    break;
	}
      }
      for(int j = 0; j<8; j++){
	if(instream >> x){
	  hf.at(i).at(j) = x;
	}
	else
	  break;
      }
    }
  }
  input();
}


// takes hcal and ecal digi input, including HF
void L1RCT::digiInput(const EcalTrigPrimDigiCollection& ecalCollection,
		      const HcalTrigPrimDigiCollection& hcalCollection)
{
  // fills input vectors with 0's in case ecal or hcal collection not used
  for (int i = 0; i < 18; i++)
    {
      for (int j = 0; j < 7; j++)
	{
	  for (int k = 0; k < 64; k++)
	    {
	      barrel.at(i).at(j).at(k) = 0;
	    }
	}
      for (int j = 0; j < 8; j++)
	{
	  hf.at(i).at(j) = 0;
	}
    }

  int nEcalDigi = ecalCollection.size();
  if (nEcalDigi>4032) {nEcalDigi=4032;}
  for (int i = 0; i < nEcalDigi; i++){
    short ieta = (short) ecalCollection[i].id().ieta(); 
    // Note absIeta counts from 1-28 (not 0-27)
    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short cal_iphi = (unsigned short) ecalCollection[i].id().iphi(); 
    unsigned short iphi = (72 + 18 - cal_iphi) % 72; // transform TOWERS (not regions) into local rct (intuitive) phi bins

    //map digis to crates, cards, and towers
    unsigned short crate = 999, card = 999, tower = 999;
    crate = rctLookupTables_->rctParameters()->calcCrate(iphi, ieta);
    card = rctLookupTables_->rctParameters()->calcCard(iphi, absIeta);
    tower = rctLookupTables_->rctParameters()->calcTower(iphi, absIeta);

    unsigned short energy = ecalCollection[i].compressedEt();
    unsigned short fineGrain = (unsigned short) ecalCollection[i].fineGrain();  // 0 or 1
    unsigned short ecalInput = energy*2 + fineGrain;

    // put input into correct crate/card/tower of barrel
    if ((crate<18) && (card<7) && (tower<32)) {             // changed 64 to 32 Sept. 19 J. Leonard, rm -1 7Nov07
      barrel.at(crate).at(card).at(tower) = ecalInput;        // rm -1
    }
    else { std::cerr << "L1RCT: ecal out of range! tower = " << tower << " iphi is " << iphi << " absieta is " << absIeta << std::endl; }
  }

//same for hcal, once we get the hcal digis, just need to add 32 to towers:
// just copied and pasted and changed names where necessary
  int nHcalDigi = hcalCollection.size();
  //if (nHcalDigi != 4176){ std::cout << "L1RCT: Warning: There are " << nHcalDigi << "hcal digis instead of 4176!" << std::endl;}
  // incl HF 4032 + 144 = 4176
  for (int i = 0; i < nHcalDigi; i++){
    short ieta = (short) hcalCollection[i].id().ieta(); 
    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short cal_iphi = (unsigned short) hcalCollection[i].id().iphi();
    // All Hcal primitives (including HF) are reported
    // with phi bin numbering in the range 0-72.
    unsigned short iphi = (72 + 18 - cal_iphi) % 72;
    // transform Hcal TOWERS (1-72)into local rct (intuitive) phi bins (72 bins) 0-71
    // Use local iphi to work out the region and crate (for HB/HE and HF)
    // HF regions need to have local iphi 0-17
    if (absIeta >= 29) {
      iphi = iphi/4;
    }

    //map digis to crates, cards, and towers
    unsigned short crate = 999, card = 999, tower = 999;
    crate = rctLookupTables_->rctParameters()->calcCrate(iphi, ieta);
    if (absIeta < 29){
      card = rctLookupTables_->rctParameters()->calcCard(iphi, absIeta);
    }
    tower = rctLookupTables_->rctParameters()->calcTower(iphi, absIeta);


    unsigned short energy = hcalCollection[i].SOI_compressedEt();     // access only sample of interest
    unsigned short fineGrain = (unsigned short) hcalCollection[i].SOI_fineGrain();
    unsigned short hcalInput = energy*2 + fineGrain;
    if (absIeta <= 28){
      // put input into correct crate/card/tower of barrel
      if ((crate<18) && (card<7) && (tower<32)) {               // changed 64 to 32 Sept. 19 J. Leonard, rm -1 7Nov07
        barrel.at(crate).at(card).at(tower + 32) = hcalInput;  // hcal towers are ecal + 32 see RC.cc; rm -1 7Nov07
      }
      else { std::cout << "L1RCT: hcal out of range!  tower = " << tower << std::endl; }
    }
    else if ((absIeta >= 29) && (absIeta <= 32)){
      // put input into correct crate/region of HF
      if ((crate<18) && (tower<8)) {
        hf.at(crate).at(tower) = hcalInput;
      }
      else { std::cout << "L1RCT: hf out of range!  region = " << tower << std::endl; }
    }

  }
  
  input();

  return;

}

//As the name implies, it will randomly generate input for the 
//regional calotrigger.
void L1RCT::randomInput()
{
  for(int i = 0; i<18;i++){
    for(int j = 0; j<7;j++){
      for(int k = 0; k<64; k++){
	barrel.at(i).at(j).at(k) = rand()%511;
      }
    }
    for(int j = 0; j<8;j++){
      hf.at(i).at(j) = rand()%255;  // changed from 1023 (10 bits)
    }
  }
  input();
  return;
}


//This method handles the bulk of the pointer passing, giving
//to each region pointers to its neighbors.  If it does *not*
//have a neighbor in that direction then it passes it a pointer
//to an empty region that contains no data and is disconnected
//from anything else.  This makes the electron finding algorithm simpler
//as then all regions can be treated equally.
void L1RCT::shareNeighbors(){
  L1RCTRegion *north;
  L1RCTRegion *south;
  L1RCTRegion *west;
  L1RCTRegion *east;
  L1RCTRegion *se;
  L1RCTRegion *sw;
  L1RCTRegion *nw;
  L1RCTRegion *ne;
  L1RCTRegion *primary;


  for(int i = 0; i < 18; i++){
    for(int j = 0; j < 7; j++){
      for(int k = 0; k < 2; k++){

	primary = crates.at(i).getReceiverCard(j)->getRegion(k);

	vector<int> northIndices = neighborMap.north(i,j,k);
	if(northIndices.at(0) != -1)
	  north = crates.at(northIndices.at(0)).getReceiverCard(northIndices.at(1))->getRegion(northIndices.at(2));
	else north = &empty;

	vector<int> southIndices = neighborMap.south(i,j,k);
	if(southIndices.at(0) != -1)
	  south = crates.at(southIndices.at(0)).getReceiverCard(southIndices.at(1))->getRegion(southIndices.at(2));
	else south = &empty;

	vector<int> westIndices = neighborMap.west(i,j,k);
	if(westIndices.at(0) != -1)
	  west = crates.at(westIndices.at(0)).getReceiverCard(westIndices.at(1))->getRegion(westIndices.at(2));
	else west = &empty;

	vector<int> eastIndices = neighborMap.east(i,j,k);
	if(eastIndices.at(0) != -1)
	  east = crates.at(eastIndices.at(0)).getReceiverCard(eastIndices.at(1))->getRegion(eastIndices.at(2));
	else east = &empty;

	vector<int> seIndices = neighborMap.se(i,j,k);
	if(seIndices.at(0) != -1)
	  se = crates.at(seIndices.at(0)).getReceiverCard(seIndices.at(1))->getRegion(seIndices.at(2));
	else se = &empty;

	vector<int> swIndices = neighborMap.sw(i,j,k);
	if(swIndices.at(0) != -1)
	  sw= crates.at(swIndices.at(0)).getReceiverCard(swIndices.at(1))->getRegion(swIndices.at(2));
	else sw = &empty;

	vector<int> neIndices = neighborMap.ne(i,j,k);
	if(neIndices.at(0) != -1)
	  ne= crates.at(neIndices.at(0)).getReceiverCard(neIndices.at(1))->getRegion(neIndices.at(2));
	else ne = &empty;

	vector<int> nwIndices = neighborMap.nw(i,j,k);
	if(nwIndices.at(0) != -1)
	  nw= crates.at(nwIndices.at(0)).getReceiverCard(nwIndices.at(1))->getRegion(nwIndices.at(2));
	else nw = &empty;

	primary->setNorthEt(north->giveNorthEt());
	primary->setNorthHE_FG(north->giveNorthHE_FG());
	primary->setSouthEt(south->giveSouthEt());
	primary->setSouthHE_FG(south->giveSouthHE_FG());
	primary->setEastEt(east->giveEastEt());
	primary->setEastHE_FG(east->giveEastHE_FG());
	primary->setWestEt(west->giveWestEt());
	primary->setWestHE_FG(west->giveWestHE_FG());
	primary->setSEEt(se->giveSEEt());
	primary->setSEHE_FG(se->giveSEHE_FG());
	primary->setSWEt(sw->giveSWEt());
	primary->setSWHE_FG(sw->giveSWHE_FG());
	primary->setNWEt(nw->giveNWEt());
	primary->setNWHE_FG(nw->giveNWHE_FG());
	primary->setNEEt(ne->giveNEEt());
	primary->setNEHE_FG(ne->giveNEHE_FG());



      }
    }
  }

}

void L1RCT::print(){
  for(int i = 0; i<18; i++){
    std::cout << "Crate " << i << std::endl;
    crates.at(i).print();
  } 
}

// Returns the top four isolated electrons from given crate
// in a vector of L1CaloEmCands
L1CaloEmCollection L1RCT::getIsolatedEGObjects(unsigned crate){
  std::vector<unsigned short> isoEmObjects = crates.at(crate).getIsolatedEGObjects();
  L1CaloEmCollection isoEmCands;
  for (uint16_t i = 0; i < 4; i++){
    unsigned rgn = ((isoEmObjects.at(i)) & 1);
    unsigned crd = (((isoEmObjects.at(i))/2) & 7);
    unsigned energy = ((isoEmObjects.at(i))/16);
    unsigned rank = rctLookupTables_->emRank(energy);
    L1CaloEmCand isoCand(rank, rgn, crd, crate, 1, i, 0);  // includes emcand index
    isoEmCands.push_back(isoCand);
  }
  return isoEmCands;
}


// Returns the top four nonisolated electrons from the given crate
// in a vector of L1CaloEmCands
L1CaloEmCollection L1RCT::getNonisolatedEGObjects(unsigned crate){
  std::vector<unsigned short> nonIsoEmObjects = crates.at(crate).getNonisolatedEGObjects();
  L1CaloEmCollection nonIsoEmCands;
  for (uint16_t i = 0; i < 4; i++){
    unsigned rgn = ((nonIsoEmObjects.at(i)) & 1);
    unsigned crd = (((nonIsoEmObjects.at(i))/2) & 7);
    unsigned energy = ((nonIsoEmObjects.at(i))/16);
    unsigned rank = rctLookupTables_->emRank(energy);
    L1CaloEmCand nonIsoCand(rank, rgn, crd, crate, 0, i, 0);  // includes emcand index
    nonIsoEmCands.push_back(nonIsoCand);
  }
  return nonIsoEmCands;
}

vector<L1CaloRegion> L1RCT::getRegions(unsigned crate){
  // barrel regions
  std::bitset<14> taus( (long) crates.at(crate).getTauBits());
  std::bitset<14> mips( (long) crates.at(crate).getMIPBits());
  std::bitset<14> quiets( (long) crates.at(crate).getQuietBits());
  std::bitset<14> overflows( (long) crates.at(crate).getOverFlowBits());
  std::vector<unsigned short> barrelEnergies = crates.at(crate).getBarrelRegions();
  std::vector<L1CaloRegion> regionCollection;
  for (unsigned card = 0; card < 7; card++){
    for (unsigned rgn = 0; rgn < 2; rgn++){
      bool tau = taus[card*2+rgn];
      bool mip = mips[card*2+rgn];
      bool quiet = quiets[card*2+rgn];
      bool overflow = overflows[card*2+rgn];
      unsigned barrelEnergy = barrelEnergies.at(card*2+rgn);
      L1CaloRegion region(barrelEnergy, overflow, tau, mip, quiet, crate, card, rgn);
      regionCollection.push_back(region);
    }
  }

  // hf regions
  std::vector<unsigned short> hfEnergies = crates.at(crate).getHFRegions();
  // fine grain bits -- still have to work out digi input
  std::vector<unsigned short> hfFineGrainBits = crates.at(crate).getHFFineGrainBits();
  for (unsigned hfRgn = 0; hfRgn<8; hfRgn++){  // region number, see diagram on paper.  make sure know how hf regions come in. 
    unsigned energy = hfEnergies.at(hfRgn);
    bool fineGrain = hfFineGrainBits.at(hfRgn);
    L1CaloRegion hfRegion(energy, fineGrain, crate, hfRgn);
    regionCollection.push_back(hfRegion);
  }
  return regionCollection;
}
