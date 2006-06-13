#include "L1RCT.h"

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

L1RCT::L1RCT() : neighborMap(),empty(){
  for(int i = 0; i<18; i++){
    L1RCTCrate c(i);
    crates.push_back(c);
  }
}

void L1RCT::input(vector<vector<vector<unsigned short> > > barrel,
		  vector<vector<unsigned short> > hf){
  for(int i = 0; i<18; i++)
    crates.at(i).input(barrel.at(i),hf.at(i));
}

//This is a method for taking input from a file.  Any entries in excess
//of 18*7*64 will simply be ignored.  This *only* fills input for a single
//event.  At the moment you cannot put a ton of data and have it be
//read as seperate events.
void L1RCT::fileInput(char* filename){
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,vector<unsigned short>(64)));
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
  unsigned short x;
  std::ifstream instream(filename);
  if(instream){
    for(int i = 0; i<18;i++){
      for(int j = 0; j<7; j++){
	for(int k = 0; k<64; k++){
	  if(instream >> x)
	    barrel.at(i).at(j).at(k) = x;
	  else
	    break;
	}
      }
      for(int j = 0; j<8; j++){
	if(instream >> x)
	  hf.at(i).at(j) = x;
	else
	  break;
      }
    }
  }
  input(barrel,hf);
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
      hf.at(i).at(j) = rand()%1023;
    }
  }
  
  input(barrel,hf);
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
