#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

#include <vector>
using std::vector;

#include <fstream>
#include <string>

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>

//#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

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
    L1RCTCrate c(i);
    crates.push_back(c);
  }
}

L1RCT::L1RCT(std::string lutFile) : empty(),neighborMap(){
  lut = new L1RCTLookupTables(lutFile);
  makeCrates();
}

L1RCT::L1RCT(std::string lutFile, 
	     std::string lutFile2,
	     std::string rctTestInputFile,
	     std::string rctTestOutputFile,
	     bool patternTest) : 
  empty(),
  neighborMap(),
  rctTestInputFile_(rctTestInputFile),
  rctTestOutputFile_(rctTestOutputFile),
  patternTest_(patternTest)
{
  lut = new L1RCTLookupTables(lutFile, lutFile2, patternTest_);
  makeCrates();
}

L1RCT::L1RCT(std::string lutFile, 
	     edm::ESHandle<CaloTPGTranscoder> transcoder, 
	     std::string rctTestInputFile, 
	     std::string rctTestOutputFile) : 
  empty(),
  neighborMap(),
  rctTestInputFile_(rctTestInputFile),
  rctTestOutputFile_(rctTestOutputFile)  
{
  transcoder_ = transcoder;
  patternTest_ = false;
  lut = new L1RCTLookupTables(lutFile, transcoder);
  makeCrates();
  patternTest_ = false;
}

void L1RCT::setGctEmScale(const L1CaloEtScale* scale){
  gctEmScale = scale;
}

void L1RCT::input(vector<vector<vector<unsigned short> > > barrel,
		  vector<vector<unsigned short> > hf){
  for(int i = 0; i<18; i++){
    crates.at(i).input(barrel.at(i),hf.at(i),lut);
  }
}

//This is a method for taking input from a file.  Any entries in excess
//of 18*7*64 will simply be ignored.  This *only* fills input for a single
//event.  At the moment you cannot put a ton of data and have it be
//read as separate events.
void L1RCT::fileInput(const char* filename){            // added "const" also in .h
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,vector<unsigned short>(64)));
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
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
  input(barrel,hf);
}


// takes hcal and ecal digi input, including HF
void L1RCT::digiInput(EcalTrigPrimDigiCollection ecalCollection, HcalTrigPrimDigiCollection hcalCollection){
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,vector<unsigned short>(64)));
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));

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
    crate = calcCrate(iphi, ieta);
    card = calcCard(iphi, absIeta);
    tower = calcTower(iphi, absIeta);

    unsigned short energy = ecalCollection[i].compressedEt();
    unsigned short fineGrain = (unsigned short) ecalCollection[i].fineGrain();  // 0 or 1
    unsigned short ecalInput = energy*2 + fineGrain;

    // put input into correct crate/card/tower of barrel
    if ((crate<18) && (card<7) && ((tower - 1)<32)) {             // changed 64 to 32 Sept. 19 J. Leonard
      barrel.at(crate).at(card).at(tower - 1) = ecalInput;        // 
    }
    else { std::cerr << "L1RCT: out of range!"; }
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
    crate = calcCrate(iphi, ieta);
    if (absIeta < 29){
      card = calcCard(iphi, absIeta);
    }
    tower = calcTower(iphi, absIeta);

    unsigned short energy = hcalCollection[i].SOI_compressedEt();     // access only sample of interest
    unsigned short fineGrain = (unsigned short) hcalCollection[i].SOI_fineGrain();
    unsigned short hcalInput = energy*2 + fineGrain;
    if (absIeta <= 28){
      // put input into correct crate/card/tower of barrel
      if ((crate<18) && (card<7) && ((tower - 1)<32)) {               // changed 64 to 32 Sept. 19 J. Leonard
        barrel.at(crate).at(card).at(tower - 1 + 32) = hcalInput;  // hcal towers are ecal + 32 see RC.cc
      }
      else { std::cout << "L1RCT: hcal out of range!"; }
    }
    else if ((absIeta >= 29) && (absIeta <= 32)){
      // put input into correct crate/region of HF
      if ((crate<18) && (tower<8)) {
        hf.at(crate).at(tower) = hcalInput;
      }
      else { std::cout << "L1RCT: hf out of range!"; }
    }
  }
  
  input(barrel,hf);

  saveRCTInput(barrel);

  return;

}

// Create file for hardware playback when requested
void L1RCT::saveRCTInput(vector<vector<vector<unsigned short> > > barrel)
{
  // Mike and Kira -- this is your playpen
  if(strcmp(rctTestInputFile_.c_str(), "-NONE-") == 0) return;
  static std::ofstream file_out(rctTestInputFile_.c_str(), std::ios::app);
  if(!file_out)
    {
      std::cerr << "Could not create " << rctTestInputFile_ << endl;
      exit(1);
    }
  static int event = 0;
  if(event == 0)
    {
      file_out
	<< "Crate = 0-17" << std::endl
	<< "Card = 0-7 within the crate" << std::endl
	<< "Tower = 0-32 covers 4 x 8 covered by the card" << std::endl
	<< "EMAddr(0:8) = EMFGBit(0:0)+CompressedEMET(1:8)" << std::endl
	<< "HDAddr(0:8) = HDFGBit(0:0)+CompressedHDET(1:8) - note: HDFGBit(0:0) is not part of the hardware LUT address" << std::endl
	<< "LutOut(0:17)= LinearEMET(0:6)+HoEFGVetoBit(7:7)+LinearJetET(8:16)+ActivityBit(17:17)" << std::endl
	<< "Event" << "\t"
	<< "Crate" << "\t"
	<< "Card" << "\t"
	<< "Tower" << "\t"
	<< "EMAddr" << "\t"
	<< "HDAddr" << "\t"
	<< "LUTOut"
	<< std::endl;
    }
  if(event < 64)
    {
      for(unsigned short iCrate = 0; iCrate < 18; iCrate++)
	{
	  for(unsigned short iCard = 0; iCard < 7; iCard++)
	    {
	      for(unsigned short iTower = 0; iTower < 32; iTower++)
		{
		  unsigned short ecal = barrel[iCrate][iCard][iTower] / 2;
		  unsigned short hcal = barrel[iCrate][iCard][iTower+32] / 2;
		  unsigned short fgbit = barrel[iCrate][iCard][iTower] & 1;
		  unsigned long lutOutput = lut->lookup(ecal, hcal, fgbit, iCrate, iCard, iTower);
		  file_out
		    << std::hex 
		    << event << "\t"
		    << iCrate << "\t"
		    << iCard << "\t"
		    << iTower << "\t"
		    << barrel[iCrate][iCard][iTower] << "\t"
		    << barrel[iCrate][iCard][iTower+32] << "\t"
		    << lutOutput
		    << std::dec 
		    << std::endl;
		  
		}
	    }
	}
    }
  event++;
  return;
}

//As the name implies, it will randomly generate input for the 
//regional calotrigger.
void L1RCT::randomInput(){
  
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,vector<unsigned short>(64)));
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
  
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
  
  input(barrel,hf);
  return;
}


//This method handles the bulk of the pointer passing, giving
//to each region pointers to its neighbors.  If it does *not*
//have a neighbor in that direction then it passes it a pointer
//to an empty region that contains no data and is disconnected
//from anything else.  This makes the electron finding algorithm simpler
//as then all regions can be treated equally.
void L1RCT::shareNeighbors(){
  L1RCTRegion* north;
  L1RCTRegion* south;
  L1RCTRegion* west;
  L1RCTRegion* east;
  L1RCTRegion* se;
  L1RCTRegion* sw;
  L1RCTRegion* nw;
  L1RCTRegion* ne;
  vector<vector<vector<L1RCTRegion*> > > regions(18,7);
  vector<L1RCTRegion*> rpair(2);
  for(int i = 0; i < 18; i++){
    for(int j = 0; j < 7; j++){
      for(int k = 0; k < 2; k++)
	rpair.at(k) = crates.at(i).getReceiverCard(j)->getRegion(k);
      regions.at(i).at(j) = rpair;
    }
  }
  for(int i = 0; i < 18; i++){
    for(int j = 0; j < 7; j++){
      for(int k = 0; k < 2; k++){
	vector<int> northIndices = neighborMap.north(i,j,k);
	if(northIndices.at(0) != -1)
	  north = regions.at(northIndices.at(0)).at(northIndices.at(1)).at(northIndices.at(2));
	else north = &empty;
	vector<int> southIndices = neighborMap.south(i,j,k);
	if(southIndices.at(0) != -1)
	  south = regions.at(southIndices.at(0)).at(southIndices.at(1)).at(southIndices.at(2));
	else south = &empty;
	vector<int> westIndices = neighborMap.west(i,j,k);
	if(westIndices.at(0) != -1)
	  west = regions.at(westIndices.at(0)).at(westIndices.at(1)).at(westIndices.at(2));
	else west = &empty;
	vector<int> eastIndices = neighborMap.east(i,j,k);
	if(eastIndices.at(0) != -1)
	  east = regions.at(eastIndices.at(0)).at(eastIndices.at(1)).at(eastIndices.at(2));
	else east = &empty;
	vector<int> seIndices = neighborMap.se(i,j,k);
	if(seIndices.at(0) != -1)
	  se = regions.at(seIndices.at(0)).at(seIndices.at(1)).at(seIndices.at(2));
	else se = &empty;
	vector<int> swIndices = neighborMap.sw(i,j,k);
	if(swIndices.at(0) != -1)
	  sw = regions.at(swIndices.at(0)).at(swIndices.at(1)).at(swIndices.at(2));
	else sw = &empty;
	vector<int> neIndices = neighborMap.ne(i,j,k);
	if(neIndices.at(0) != -1)
	  ne = regions.at(neIndices.at(0)).at(neIndices.at(1)).at(neIndices.at(2));
	else ne = &empty;
	vector<int> nwIndices = neighborMap.nw(i,j,k);
	if(nwIndices.at(0) != -1)
	  nw = regions.at(nwIndices.at(0)).at(nwIndices.at(1)).at(nwIndices.at(2));
	else nw = &empty;
	regions.at(i).at(j).at(k)->setNorthEt(north->giveNorthEt());
	regions.at(i).at(j).at(k)->setNorthHE_FG(north->giveNorthHE_FG());
	regions.at(i).at(j).at(k)->setSouthEt(south->giveSouthEt());
	regions.at(i).at(j).at(k)->setSouthHE_FG(south->giveSouthHE_FG());
	regions.at(i).at(j).at(k)->setEastEt(east->giveEastEt());
	regions.at(i).at(j).at(k)->setEastHE_FG(east->giveEastHE_FG());
	regions.at(i).at(j).at(k)->setWestEt(west->giveWestEt());
	regions.at(i).at(j).at(k)->setWestHE_FG(west->giveWestHE_FG());
	regions.at(i).at(j).at(k)->setSEEt(se->giveSEEt());
	regions.at(i).at(j).at(k)->setSEHE_FG(se->giveSEHE_FG());
	regions.at(i).at(j).at(k)->setSWEt(sw->giveSWEt());
	regions.at(i).at(j).at(k)->setSWHE_FG(sw->giveSWHE_FG());
	regions.at(i).at(j).at(k)->setNWEt(nw->giveNWEt());
	regions.at(i).at(j).at(k)->setNWHE_FG(nw->giveNWHE_FG());
	regions.at(i).at(j).at(k)->setNEEt(ne->giveNEEt());
	regions.at(i).at(j).at(k)->setNEHE_FG(ne->giveNEHE_FG());
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

// maps rct iphi, ieta of tower to crate
unsigned short L1RCT::calcCrate(unsigned short rct_iphi, short ieta){
  unsigned short crate = rct_iphi/8;
  if(abs(ieta) > 28) crate = rct_iphi / 2;
  if (ieta > 0){
    crate = crate + 9;
  }
  return crate;
}

//map digi rct iphi, ieta to card
unsigned short L1RCT::calcCard(unsigned short rct_iphi, unsigned short absIeta){
  unsigned short card = 999;
  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    card =  ((absIeta-1)/8)*2 + (rct_iphi%8)/4;
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    card = 6;
  }
  else{}
  return card;
}

//map digi rct iphi, ieta to tower
unsigned short L1RCT::calcTower(unsigned short rct_iphi, unsigned short absIeta){
  unsigned short tower = 999;
  unsigned short iphi = rct_iphi;
  unsigned short regionPhi = (iphi % 8)/4;

  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    tower = ((absIeta-1)%8)*4 + (iphi%4) + 1;       // assume iphi between 0 and 71; makes towers from 1-32
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    if (regionPhi == 0){
      tower = (absIeta-25)*4 + (iphi%4) + 1;   // towers from 1-32, modified Aug. 1 Jessica Leonard
    }
    else {
      tower = 29 + iphi % 4 + (25 - absIeta) * 4;
    }
  }
  // absIeta >= 29 (HF regions)
  else if ((absIeta >= 29) && (absIeta <= 32)){
    regionPhi = iphi % 2;  // SPECIAL DEFINITION OF REGIONPHI FOR HF SINCE HF IPHI IS 0-17 Sept. 19 J. Leonard
    // HF MAPPING, just regions now, don't need to worry about towers -- just calling it "tower" for convenience
    tower = (regionPhi) * 4 + absIeta - 29;
  }
  return tower;
}

short L1RCT::calcIEta(unsigned short iCrate, unsigned short iCard, unsigned short iTower)
{
  unsigned short absIEta;
  if(iCard < 6) 
    absIEta = (iCard / 2) * 8 + ((iTower - 1) / 4) + 1;
  else if(iCard == 6) {
    if(iTower < 17)
      absIEta = 25 + (iTower - 1) / 4;
    else
      absIEta = 28 - ((iTower - 17) / 4);
  }
  else
    absIEta = 29 + iTower % 4;
  short iEta;
  if(iCrate < 9) iEta = -absIEta;
  else iEta = absIEta;
  return iEta;
}

unsigned short L1RCT::calcIPhi(unsigned short iCrate, unsigned short iCard, unsigned short iTower)
{
  short iPhi;
  if(iCard < 6)
    iPhi = (iCrate % 9) * 8 + (iCard % 2) * 4 + ((iTower - 1) % 4);
  else if(iCard == 6){
    if(iTower < 17)
      iPhi = (iCrate % 9) * 8 + ((iTower - 1) % 4);
    else
      iPhi = (iCrate % 9) * 8 + ((iTower - 17) % 4) + 4;
  }
  else
    iPhi = (iCrate % 9) * 2 + iTower / 4;
  return iPhi;
}

// Returns the top four isolated electrons from given crate
// in a vector of L1CaloEmCands
L1CaloEmCollection L1RCT::getIsolatedEGObjects(unsigned crate){
  vector<unsigned short> isoEmObjects = crates.at(crate).getIsolatedEGObjects();
  L1CaloEmCollection isoEmCands;
  for (uint16_t i = 0; i < 4; i++){
    unsigned rgn = ((isoEmObjects.at(i)) & 1);
    unsigned crd = (((isoEmObjects.at(i))/2) & 7);
    unsigned short energy = ((isoEmObjects.at(i))/16);
    unsigned rank = gctEmScale->rank(energy);
    L1CaloEmCand isoCand(rank, rgn, crd, crate, 1, i, 0);  // includes emcand index
    isoEmCands.push_back(isoCand);
  }
  return isoEmCands;
}


// Returns the top four nonisolated electrons from the given crate
// in a vector of L1CaloEmCands
L1CaloEmCollection L1RCT::getNonisolatedEGObjects(unsigned crate){
  vector<unsigned short> nonIsoEmObjects = crates.at(crate).getNonisolatedEGObjects();
  L1CaloEmCollection nonIsoEmCands;
  for (uint16_t i = 0; i < 4; i++){
    unsigned rgn = ((nonIsoEmObjects.at(i)) & 1);
    unsigned crd = (((nonIsoEmObjects.at(i))/2) & 7);
    unsigned short energy = ((nonIsoEmObjects.at(i))/16);
    unsigned rank = gctEmScale->rank(energy);
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
  vector<unsigned short> barrelEnergies = crates.at(crate).getBarrelRegions();
  vector<L1CaloRegion> regionCollection;
  for (unsigned card = 0; card < 7; card++){
    for (unsigned rgn = 0; rgn < 2; rgn++){
      bool tau = taus[card*2+rgn];
      bool mip = mips[card*2+rgn];
      bool quiet = quiets[card*2+rgn];
      bool overflow = overflows[card*2+rgn];
      unsigned barrelEnergy = barrelEnergies.at(card*2+rgn);
      L1CaloRegion region(barrelEnergy, overflow, tau, mip, quiet, crate, card, rgn); // change if necessary
      regionCollection.push_back(region);
    }
  }
  // hf regions
  vector<unsigned short> hfEnergies = crates.at(crate).getHFRegions();
  // fine grain bits -- still have to work out digi input
  vector<unsigned short> hfFineGrainBits = crates.at(crate).getHFFineGrainBits();
  for (unsigned hfRgn = 0; hfRgn<8; hfRgn++){  // region number, see diagram on paper.  make sure know how hf regions come in. 
    unsigned energy = hfEnergies.at(hfRgn);
    bool fineGrain = hfFineGrainBits.at(hfRgn);
    L1CaloRegion hfRegion(energy, fineGrain, crate, hfRgn);
    regionCollection.push_back(hfRegion);
  }
  return regionCollection;
}
