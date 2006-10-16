#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include <fstream>

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

L1RCT::L1RCT() : empty(),neighborMap(){
  for(int i = 0; i<18; i++){
    L1RCTCrate c(i);
    crates.push_back(c);
  }
}

void L1RCT::input(vector<vector<vector<unsigned short> > > barrel,
		  vector<vector<unsigned short> > hf){
  //cout << "L1RCT::input() entered" << endl;
  for(int i = 0; i<18; i++){
    //cout << "calling Crate.input() for crate " << i << endl;
    crates.at(i).input(barrel.at(i),hf.at(i));
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
  //cout << "L1RCT::fileInput() entered" << endl;
  std::ifstream instream(filename);
  if(instream){
    //cout << "file opened in L1RCT::fileInput()" << endl;
    for(int i = 0; i<18;i++){
      for(int j = 0; j<7; j++){
	for(int k = 0; k<64; k++){
	  if(instream >> x){
	    unsigned short bit = x/256;             // added J.Leonard Aug. 16 06
	    unsigned short energy = x&255;          //
	    unsigned short input = energy*2 + bit;  //
	    //	    barrel.at(i).at(j).at(k) = x;
	    barrel.at(i).at(j).at(k) = input;
	    //cout << x;
	  }
	  else
	    break;
	}
      }
      for(int j = 0; j<8; j++){
	if(instream >> x){
	  hf.at(i).at(j) = x;
	  //cout << x;
	}
	else
	  break;
      }
    }
    //cout << "input filled from file" << endl;
  }
  //cout << "calling L1RCT::input()" << endl;
  input(barrel,hf);
  //cout << "L1RCT::input() called" << endl;
}


// takes hcal and ecal digi input, including HF
void L1RCT::digiInput(EcalTrigPrimDigiCollection ecalCollection, HcalTrigPrimDigiCollection hcalCollection){
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,vector<unsigned short>(64)));
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
  //unsigned short x;
  vector<vector<unsigned short> > ecalBarrel(72,vector<unsigned short>(56));
  vector<vector<unsigned short> > hcalBarrel(72,vector<unsigned short>(56));
  //vector<vector<unsigned short> > hcalForward(18,vector<unsigned short>(8));

  std::ofstream file_out("towerinput.txt", std::ios::app);
  if (!file_out){
    std::cerr << "Tower input file did not open!" << endl;
    return;
  }

// ecal:
//  cout << "\n\nECAL" << endl;
//  cout << "\t\t\t\t\tCrate\tCard\tTower\tInput" << endl;
  int nEcalDigi = ecalCollection.size();
  if (nEcalDigi>4032) {nEcalDigi=4032;}
  for (int i = 0; i < nEcalDigi; i++){
    short ieta = (short) ecalCollection[i].id().ieta(); 
//     if (ecalCollection[i].compressedEt()>0) { 
//    cout << "Energy " << ecalCollection[i].compressedEt()
//	   <<" eta " << ieta; 
//     }
    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short cal_iphi = (unsigned short) ecalCollection[i].id().iphi(); 
//     if (ecalCollection[i].compressedEt()>0) { 
//    cout << " raw phi " << iphi ; 
//     }
    unsigned short iphi = (72 + 20 - cal_iphi) % 72;         //    transform TOWERS (not regions) into local rct (intuitive) phi bins
//     if (ecalCollection[i].compressedEt()>0) { 
//    cout << " rct phi " << iphi << "  "; 
//     }
//    unsigned short regionPhi = (iphi % 8)/4;

    //map digis to crates, cards, and towers
    unsigned short crate = 999, card = 999, tower = 999;
//    crate = iphi/8;
//    if (ieta > 0){
//      crate = crate + 9;
//    }

    crate = calcCrate(iphi, ieta);

    // Note absIeta counts from 1-28 (not 0-27)
//    if (absIeta <= 24){
//      card  = ((absIeta-1)/8)*2 + regionPhi;      // slick integer division
//      tower = ((absIeta-1)%8)*4 + (iphi%4) + 1;   // towers from 1-32, modified Aug. 1 Jessica Leonard
//    }
//    // absIeta >= 25
//    else {
//      card = 6;
//      if (regionPhi == 0){
//	tower = (absIeta-25)*4 + (iphi%4) + 1;   // towers from 1-32, modified Aug. 1 Jessica Leonard
//      }
//      else {
////	tower = (absIeta-21)*4 + (iphi%4);          // Greg's line
//        tower = (28 - absIeta)*4 + (iphi%4) + 17;   // Jessica's line
//      }
//    }

    card = calcCard(iphi, absIeta);
    tower = calcTower(iphi, absIeta);

    unsigned short energy = ecalCollection[i].compressedEt();
    unsigned short fineGrain = (unsigned short) ecalCollection[i].fineGrain();  // 0 or 1
    unsigned short ecalInput = energy*2 + fineGrain;

    // for file diagram of digi inputs
    if (ieta > 0){
      ecalBarrel.at(iphi).at(ieta - 1) = ecalInput;
      //cout << "Ecal\tiphi: " << iphi << "\tieta: " << ieta;
    }
    else {
      ecalBarrel.at(iphi).at(56 + ieta) = ecalInput;
      //cout << "Ecal\tiphi: " << iphi << "\tieta: " << ieta << "\tieta+56: " << (ieta+56);
    }

    // put input into correct crate/card/tower of barrel
    if ((crate<18) && (card<7) && ((tower - 1)<32)) {             // changed 64 to 32 Sept. 19 J. Leonard
      barrel.at(crate).at(card).at(tower - 1) = ecalInput;        // 
    }
    else { cout << "out of range!"; }
    cout << "Ecal:\t" <<crate << "\t" << card << "\t" << tower << "\t" << ecalInput << endl;
  }

//same for hcal, once we get the hcal digis, just need to add 32 to towers:
// just copied and pasted and changed names where necessary
//  cout << "\n\nHCAL" << endl;
  cout << "\t\t\t\t\tCrate\tCard\tTower\tInput" << endl;
  int nHcalDigi = hcalCollection.size();
//  if (nHcalDigi != 4176){ cout << "There are " << nHcalDigi << " instead of 4176!" << endl;}
  cout << "There are " << nHcalDigi << " hcal digis.  There should be 4176." << endl;
//  for (int i = 0; i < 4176; i++){                        // ARE THERE 4032?? think not -- incl HF 4032 + 144 = 4176
  for (int i = 0; i < nHcalDigi; i++){
    short ieta = (short) hcalCollection[i].id().ieta(); 
    if (hcalCollection[i].SOI_compressedEt()>0) { 
      cout << "Energy " << hcalCollection[i].SOI_compressedEt()
 	 << " eta " << ieta; 
    }
    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short cal_iphi = (unsigned short) hcalCollection[i].id().iphi();
//     if (hcalCollection[i].SOI_compressedEt()>0) { 
//    cout << " raw phi " << cal_iphi; 
//     }
    // All Hcal primitives (including HF) are reported
    // with phi bin numbering in the range 0-72.
    unsigned short iphi = (72 + 18 - cal_iphi) % 72;      // transform Hcal TOWERS (1-72)into local rct (intuitive) phi bins (72 bins) 0-71
    // Use local iphi to work out the region and crate (for HB/HE and HF)
//    unsigned short regionPhi = (iphi % 8)/4;
    unsigned short crate = 999, card = 999, tower = 999;
//    unsigned short crate     = (iphi / 8);
//    if (ieta > 0){
//      crate = crate + 9;
//    }
    crate = calcCrate(iphi, ieta);

    // HF regions need to have local iphi 0-17
    if (absIeta >= 29) {
      iphi = iphi/4;
    }
//     if (hcalCollection[i].SOI_compressedEt()>0) { 
//    cout << " rct phi " << iphi << "  "; 
//     }

    //map digis to crates, cards, and towers
//    unsigned short card = 999, tower = 999;
//    if (absIeta <= 24){
//      card =  ((absIeta-1)/8)*2 + regionPhi;          // integer division again
//      tower = ((absIeta-1)%8)*4 + (iphi%4) + 1;       // assume iphi between 0 and 71; makes towers from 1-32
//    }
//    // 25 <= absIeta <= 28 (card 6)
//    else if ((absIeta >= 25) && (absIeta <= 28)){
//      card = 6;
//      if (regionPhi == 0){
//	tower = (absIeta-25)*4 + (iphi%4) + 1;   // towers from 1-32, modified Aug. 1 Jessica Leonard
//      }
//      else {
////	tower = (absIeta-21)*4 + (iphi%4);          // Greg's line
//        tower = (28 - absIeta)*4 + (iphi%4) + 17;   // Jessica's line
//      }
//    }
//    // absIeta >= 29 (HF regions)
//    else if ((absIeta >= 29) && (absIeta <= 32)){
//      // HF MAPPING, just regions now, don't need to worry about towers -- just calling it "tower" for convenience
//      // Modified by Greg to give a number between 0 and 7
////       tower = (regionPhi) * 4 + absIeta - 7;
//      tower = (regionPhi) * 4 + absIeta - 29;
//    }

    if (absIeta < 29){
      card = calcCard(iphi, absIeta);
    }
    tower = calcTower(iphi, absIeta);

    //unsigned short energy = hcalCollection[i].t0().compressedEt();  // CHANGED
    unsigned short energy = hcalCollection[i].SOI_compressedEt();     // don't have to access sample
    //unsigned short fineGrain = (unsigned short) hcalCollection[i].t0().fineGrain();  // 0 or 1  // CHANGED
    unsigned short fineGrain = (unsigned short) hcalCollection[i].SOI_fineGrain();  // don't have to access sample
    unsigned short hcalInput = energy*2 + fineGrain;

    if (absIeta <= 28){

      // for file diagram of digi inputs
      if (ieta > 0){
	hcalBarrel.at(iphi).at(ieta - 1) = hcalInput;
      }
      else {
	hcalBarrel.at(iphi).at(56 + ieta) = hcalInput;
      }

      // put input into correct crate/card/tower of barrel
      if ((crate<18) && (card<7) && ((tower - 1)<32)) {               // changed 64 to 32 Sept. 19 J. Leonard
        barrel.at(crate).at(card).at(tower - 1 + 32) = hcalInput;  // hcal towers are ecal + 32 see RC.cc
      }
      else { cout << "out of range!"; }
      cout << "Hcal:\t" << crate << "\t" << card << "\t" << tower + 32 << "\t" << hcalInput << endl;
    }

    else if ((absIeta >= 29) && (absIeta <= 32)){
      // put input into correct crate/region of HF
      if ((crate<18) && (tower<8)) {
        hf.at(crate).at(tower) = hcalInput;
      }
      else { cout << "out of range!"; }
      cout << "HF: crate " << crate << "\tregion " << tower << "\tinput " << hcalInput << endl;
    }
  }

  file_out << "iphi goes from 1-72 down rows, ieta goes from -28 to 28 across columns." << endl << endl;
  file_out << "ECAL:" << endl;
  for (int i = 0; i < 72; i++){
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << ecalBarrel.at(i).at(28+j);
    }
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << ecalBarrel.at(i).at(j);
    }
    file_out << endl;
  }
  file_out << "\n\n\n\n\n" << endl;
  file_out << "HCAL:" << endl;
  for (int i = 0; i < 72; i++){
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << hcalBarrel.at(i).at(28+j);
    }
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << hcalBarrel.at(i).at(j);
    }
    file_out << endl;
  }
  file_out << "\n\n\n\n\n" << endl;

  file_out << "HF:" << endl;
  for (int i = 0; i < 9; i++){
    for (int j = 3; j >= 0; j--){
      file_out.width(3);
      file_out << hf.at(i).at(j);
    }
    file_out << "\t\t\t";
    for (int j = 0; j <= 3; j++){
      file_out.width(3);
      file_out << hf.at(i+9).at(j);
    }
    file_out << endl;
    for (int j = 7; j >= 4; j--){
      file_out.width(3);
      file_out << hf.at(i).at(j);
    }
    file_out << "\t\t\t";
    for (int j = 4; j <= 7; j++){
      file_out.width(3);
      file_out << hf.at(i+9).at(j);
    }
    file_out << endl;
  }
  file_out << "\n\n\n\n\n" << endl;
  file_out.close();

  input(barrel,hf);
  return;
}

// takes hcal digi input for crates 0 and 1, fills rest of cal with zeros
// intended for pam's hardware test ONLY
void L1RCT::digiTestInput(HcalTrigPrimDigiCollection hcalCollection){
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,vector<unsigned short>(64)));
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
  //unsigned short x;
//  vector<vector<unsigned short> > ecalBarrel(72,vector<unsigned short>(56));
//  vector<vector<unsigned short> > hcalBarrel(72,vector<unsigned short>(56));
  //vector<vector<unsigned short> > hcalForward(18,vector<unsigned short>(8));

/*
  std::ofstream file_out("towerinput.txt", std::ios::app);
  if (!file_out){
    std::cerr << "Tower input file did not open!" << endl;
    return;
  }
*/

// ecal:

  for (int i = 0; i < 18; i++){
    for (int j = 0; j < 7; j++){
      for (int k = 0; k < 32; k++){
        barrel.at(i).at(j).at(k) = 0;
      }
    }
  }


// hcal:
//  cout << "\n\nHCAL" << endl;
  cout << "\t\t\t\t\tCrate\tCard\tTower\tInput" << endl;
  int nHcalDigi = hcalCollection.size();
  cout << "There are " << nHcalDigi << " hcal digis.  There should be 448." << endl;

  for (int i = 0; i < nHcalDigi; i++){
    short ieta = (short) hcalCollection[i].id().ieta(); 
/*
    if (hcalCollection[i].SOI_compressedEt()>0) { 
      cout << "Energy " << hcalCollection[i].SOI_compressedEt()
 	 << " eta " << ieta; 
    }
*/

    unsigned short absIeta = (unsigned short) abs(ieta);
    unsigned short cal_iphi = (unsigned short) hcalCollection[i].id().iphi();
//     if (hcalCollection[i].SOI_compressedEt()>0) { 
//    cout << " raw phi " << cal_iphi; 
//     }
    // All Hcal primitives (including HF) are reported
    // with phi bin numbering in the range 0-72.
    unsigned short iphi = (72 + 18 - cal_iphi) % 72;      // transform Hcal TOWERS (1-72)into local rct (intuitive) phi bins (72 bins) 0-71

    //map digis to crates, cards, and towers

    unsigned short crate = 999, card = 999, tower = 999;
    crate = calcCrate(iphi, ieta);
    if (crate > 1) {
      cout << "WARNING: hcal digi " << i << " is in crate " << crate << endl;
    }

    if (absIeta >= 29) {
      cout << "WARNING: hcal digi " << i << " has absIeta " << absIeta << ", greater than 28!" << endl;
    }
//     if (hcalCollection[i].SOI_compressedEt()>0) { 
//    cout << " rct phi " << iphi << "  "; 
//     }

    if (absIeta < 29){
      card = calcCard(iphi, absIeta);
    }
    tower = calcTower(iphi, absIeta);

    unsigned short energy = hcalCollection[i].SOI_compressedEt();
    unsigned short fineGrain = (unsigned short) hcalCollection[i].SOI_fineGrain();
    unsigned short hcalInput = energy*2 + fineGrain;

    if (absIeta <= 28){
/*
      // for file diagram of digi inputs
      if (ieta > 0){
	hcalBarrel.at(iphi).at(ieta - 1) = hcalInput;
      }
      else {
	hcalBarrel.at(iphi).at(56 + ieta) = hcalInput;
      }
*/
      // put input into correct crate/card/tower of barrel
      if ((crate<18) && (card<7) && ((tower - 1)<32)) {
        barrel.at(crate).at(card).at(tower - 1 + 32) = hcalInput;  // hcal towers are ecal + 32 see RC.cc
      }
      else { cout << "out of range!"; }
      cout << "Hcal:\t" << crate << "\t" << card << "\t" << tower + 32 << "\t" << hcalInput << endl;
    }

/*
    else if ((absIeta >= 29) && (absIeta <= 32)){
      // put input into correct crate/region of HF
      if ((crate<18) && (tower<8)) {
        hf.at(crate).at(tower) = hcalInput;
      }
      else { cout << "out of range!"; }
      cout << "HF: crate " << crate << "\tregion " << tower << "\tinput " << hcalInput << endl;
    }
*/

  }

  for (int i = 0; i < 18; i++){
    if (i > 1){
      for (int j = 0; j < 7; j++){
        for (int k = 0; k < 32; k++){
          barrel.at(i).at(j).at(k + 32) = 0;
          cout << "hcal barrel " << i << " " << j << " " << k << " is " << barrel.at(i).at(j).at(k+32) << endl;
        }
      }
    }
    for (int j = 0; j < 8; j++){
      hf.at(i).at(j) = 0;
      cout << "hf " << i << " " << j << " is " << hf.at(i).at(j) << endl;
    }
  }

/*
  file_out << "iphi goes from 1-72 down rows, ieta goes from -28 to 28 across columns." << endl << endl;
  file_out << "ECAL:" << endl;
  for (int i = 0; i < 72; i++){
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << ecalBarrel.at(i).at(28+j);
    }
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << ecalBarrel.at(i).at(j);
    }
    file_out << endl;
  }
  file_out << "\n\n\n\n\n" << endl;
  file_out << "HCAL:" << endl;
  for (int i = 0; i < 72; i++){
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << hcalBarrel.at(i).at(28+j);
    }
    for (int j = 0; j < 28; j++){
      file_out.width(3);
      file_out << hcalBarrel.at(i).at(j);
    }
    file_out << endl;
  }
  file_out << "\n\n\n\n\n" << endl;

  file_out << "HF:" << endl;
  for (int i = 0; i < 9; i++){
    for (int j = 3; j >= 0; j--){
      file_out.width(3);
      file_out << hf.at(i).at(j);
    }
    file_out << "\t\t\t";
    for (int j = 0; j <= 3; j++){
      file_out.width(3);
      file_out << hf.at(i+9).at(j);
    }
    file_out << endl;
    for (int j = 7; j >= 4; j--){
      file_out.width(3);
      file_out << hf.at(i).at(j);
    }
    file_out << "\t\t\t";
    for (int j = 4; j <= 7; j++){
      file_out.width(3);
      file_out << hf.at(i+9).at(j);
    }
    file_out << endl;
  }
  file_out << "\n\n\n\n\n" << endl;
  file_out.close();
*/

  input(barrel,hf);
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
    cout << "Crate " << i << endl;
    crates.at(i).print();
  } 
}

// maps rct iphi, ieta of tower to crate
unsigned short L1RCT::calcCrate(unsigned short rct_iphi, short ieta){
  unsigned short crate = rct_iphi/8;
  if (ieta > 0){
    crate = crate + 9;
  }
  return crate;
}

//map digi rct iphi, ieta to card
unsigned short L1RCT:: calcCard(unsigned short rct_iphi, unsigned short absIeta){
  unsigned short card = 999;
  unsigned short iphi = rct_iphi;
  unsigned short regionPhi = (iphi % 8)/4;

  // Note absIeta counts from 1-32 (not 0-31)
  if (absIeta <= 24){
    card =  ((absIeta-1)/8)*2 + regionPhi;          // integer division again
  }
  // 25 <= absIeta <= 28 (card 6)
  else if ((absIeta >= 25) && (absIeta <= 28)){
    card = 6;
  }
  else{}
  return card;
}

//map digi rct iphi, ieta to tower
unsigned short L1RCT:: calcTower(unsigned short rct_iphi, unsigned short absIeta){
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
      tower = (28 - absIeta)*4 + (iphi%4) + 17;
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

// Returns the top four isolated electrons from given crate
// in a vector of L1CaloEmCands
L1CaloEmCollection L1RCT::getIsolatedEGObjects(int crate){
  vector<unsigned short> isoEmObjects = crates.at(crate).getIsolatedEGObjects();
  L1CaloEmCollection isoEmCands;
  // cout << "\nCrate " << crate << endl;
  for (int i = 0; i < 4; i++){
    unsigned short rgn = ((isoEmObjects.at(i)) & 1);
    unsigned short crd = (((isoEmObjects.at(i))/2) & 7);
    unsigned short energy = ((isoEmObjects.at(i))/16);
    L1CaloEmCand isoCand(energy, rgn, crd, crate, 1);  // uses 7-bit energy as rank here, temporarily
    // cout << "card " << crd << "region " << rgn << "energy " << energy << endl;
    isoEmCands.push_back(isoCand);
  }
  return isoEmCands;
}


// Returns the top four nonisolated electrons from the given crate
// in a vector of L1CaloEmCands
L1CaloEmCollection L1RCT::getNonisolatedEGObjects(int crate){
  vector<unsigned short> nonIsoEmObjects = crates.at(crate).getNonisolatedEGObjects();
  L1CaloEmCollection nonIsoEmCands;
  for (int i = 0; i < 4; i++){
    unsigned short rgn = ((nonIsoEmObjects.at(i)) & 1);
    unsigned short crd = (((nonIsoEmObjects.at(i))/2) & 7);
    unsigned short energy = ((nonIsoEmObjects.at(i))/16);
    L1CaloEmCand nonIsoCand(energy, rgn, crd, crate, 0);  // uses 7-bit energy as rank here, temporarily
    nonIsoEmCands.push_back(nonIsoCand);
  }
  return nonIsoEmCands;
}


vector<L1CaloRegion> L1RCT::getRegions(int crate){
  // barrel regions
  bitset<14> taus( (long) crates.at(crate).getTauBits());
  bitset<14> mips( (long) crates.at(crate).getMIPBits());
  bitset<14> quiets( (long) crates.at(crate).getQuietBits());
  bitset<14> overflows( (long) crates.at(crate).getOverFlowBits());
  vector<unsigned short> barrelEnergies = crates.at(crate).getBarrelRegions();
  vector<L1CaloRegion> regionCollection;
  for (int card = 0; card < 7; card++){
    for (int rgn = 0; rgn < 2; rgn++){
      unsigned short tau = taus[card*2+rgn];
      cout << "Crate: " << crate << "\tCard: " << card << "\tRegion: " << rgn << "\tTau veto " << tau << endl;
      unsigned short mip = mips[card*2+rgn];
      unsigned short quiet = quiets[card*2+rgn];
      unsigned short overflow = overflows[card*2+rgn];
      unsigned short barrelEnergy = barrelEnergies.at(card*2+rgn);
      L1CaloRegion region(barrelEnergy, overflow, tau, mip, quiet, crate, card, rgn); // change if necessary
      regionCollection.push_back(region);
    }
  }

  // hf regions
  vector<unsigned short> hfEnergies = crates.at(crate).getHFRegions();
  // fine grain bits -- still have to work out digi input
  vector<unsigned short> hfFineGrainBits = crates.at(crate).getHFFineGrainBits();
  for (int i = 0; i<8; i++){  // region number, see diagram on paper.  make sure know how hf regions come in. 
    int hfRgn = 10;
    if (i <= 3) {
      hfRgn = 3 - i;     // rearranging index for low phi
    }
    else {
      hfRgn = 11 - i;    // rearranging index for high phi
    }
    unsigned short fineGrain = hfFineGrainBits.at(i);
    unsigned short energy = hfEnergies.at(i);
    L1CaloRegion hfRegion(energy, fineGrain, crate, i);  // no overflow  // CHANGED hfRgn to i !!  to match rest of hf mapping Sept. 22 J. Leonard
    regionCollection.push_back(hfRegion);
  }
  return regionCollection;
}
