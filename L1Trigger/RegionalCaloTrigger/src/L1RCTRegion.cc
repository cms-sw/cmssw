#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRegion.h"

L1RCTRegion::L1RCTRegion() : totalRegionEt(36),etIn9Bits(16),
			     totalRegionHE_FG(36),muonBit(16),activityBit(16)
{
}

unsigned short L1RCTRegion::getEtIn7Bits(int i, int j){
  //i & j run 0-3
  return totalRegionEt.at(6*(i+1) + j+1);
}

void L1RCTRegion::setEtIn7Bits(int i, int j,unsigned short energy){
  //i & j should be 0-3
  if(energy <= 127)
    totalRegionEt.at(6*(i+1) + j+1) = energy;
  else
    totalRegionEt.at(6*(i+1) + j+1) = 127;
}

unsigned short L1RCTRegion::getEtIn9Bits(int i, int j){
  return etIn9Bits.at(4*i + j);
}

void L1RCTRegion::setEtIn9Bits(int i, int j,unsigned short energy){
  if(energy <=511)
    etIn9Bits.at(4*i+j) = energy;
  else 
    etIn9Bits.at(4*i+j) = 511;
}

unsigned short L1RCTRegion::getHE_FGBit(int i, int j){
  return totalRegionHE_FG.at(6*(i+1)+j+1);
}

void L1RCTRegion::setHE_FGBit(int i, int j, unsigned short HE_FG){
  totalRegionHE_FG.at(6*(i+1)+j+1) = HE_FG;
}

unsigned short L1RCTRegion::getMuonBit(int i, int j){
  return muonBit.at(4*i+j);
}

void L1RCTRegion::setMuonBit(int i, int j,unsigned short muon){
  muonBit.at(4*i+j) = muon;
}

void L1RCTRegion::setActivityBit(int i, int j, unsigned short activity){
  activityBit.at(4*i+j) = activity;
}

unsigned short L1RCTRegion::getActivityBit(int i, int j){
  return activityBit.at(4*i+j);
}

vector<unsigned short> L1RCTRegion::giveNorthEt(){
  vector<unsigned short> north(4);
  for(int i = 0; i<4;i++)
    north.at(i) = getEtIn7Bits(3,i);
  return north;
}
void L1RCTRegion::setNorthEt(vector<unsigned short> north){
  for(int i = 0; i<4; i++)
    totalRegionEt.at(i+1) = north.at(i);
}
vector<unsigned short> L1RCTRegion::giveNorthHE_FG(){
  vector<unsigned short> north(4);
  for(int i = 0; i<4; i++)
    north.at(i) = getHE_FGBit(3,i);
  return north;
}
void L1RCTRegion::setNorthHE_FG(vector<unsigned short> north){
  for(int i = 0; i<4; i++)
    totalRegionHE_FG.at(i+1) = north.at(i);
}

vector<unsigned short> L1RCTRegion::giveSouthEt(){
  vector<unsigned short> south(4);
  for(int i = 0; i<4; i++)
    south.at(i) = getEtIn7Bits(0,i);
  return south;
}
void L1RCTRegion::setSouthEt(vector<unsigned short> south){
  for(int i = 0; i<4; i++)
    totalRegionEt.at(31+i) = south.at(i);
}

vector<unsigned short> L1RCTRegion::giveSouthHE_FG(){
  vector<unsigned short> south(4);
  for(int i = 0; i<4; i++)
    south.at(i) = getHE_FGBit(0,i);
  return south;
}
void L1RCTRegion::setSouthHE_FG(vector<unsigned short> south){
  for(int i=0; i<4; i++)
    totalRegionHE_FG.at(31+i) = south.at(i);
}

vector<unsigned short> L1RCTRegion::giveWestEt(){
  vector<unsigned short> west(4);
  for(int i =0; i<4; i++)
    west.at(i) = getEtIn7Bits(i,3);
  return west;
}
void L1RCTRegion::setWestEt(vector<unsigned short> west){
  for(int i = 0; i<4; i++)
    totalRegionEt.at(6*(i+1)) = west.at(i);
}

vector<unsigned short> L1RCTRegion::giveWestHE_FG(){
  vector<unsigned short> west(4);
  for(int i = 0; i<4; i++)
    west.at(i) = getHE_FGBit(i,3);
  return west;
}
void L1RCTRegion::setWestHE_FG(vector<unsigned short> west){
  for(int i = 0; i<4; i++)
    totalRegionHE_FG.at(6*(i+1)) = west.at(i);
}

vector<unsigned short> L1RCTRegion::giveEastEt(){
  vector<unsigned short> east(4);
  for(int i = 0; i<4; i++)
    east.at(i) = getEtIn7Bits(i,0);
  return east;
}
void L1RCTRegion::setEastEt(vector<unsigned short> east){
  for(int i = 0; i<4; i++)
    totalRegionEt.at(6*(i+1) + 5) = east.at(i);
}

vector<unsigned short> L1RCTRegion::giveEastHE_FG(){
  vector<unsigned short> east(4);
  for(int i = 0; i<4; i++)
    east.at(i) = getHE_FGBit(i,0);
  return east;
}
void L1RCTRegion::setEastHE_FG(vector<unsigned short> east){
  for(int i = 0; i<4; i++)
    totalRegionHE_FG.at(6*(i+1) + 5) = east.at(i);
}

unsigned short L1RCTRegion::giveNEEt(){
  return getEtIn7Bits(3,0)&7;
}
unsigned short L1RCTRegion::giveNEHE_FG(){
  return getHE_FGBit(3,0);
}
void L1RCTRegion::setNEEt(unsigned short ne){
  totalRegionEt.at(5) = ne;
}
void L1RCTRegion::setNEHE_FG(unsigned short ne){
  totalRegionHE_FG.at(5) = ne;
}

unsigned short L1RCTRegion::giveNWEt(){
  return getEtIn7Bits(3,3)&7;
}
unsigned short L1RCTRegion::giveNWHE_FG(){
  return getHE_FGBit(3,3);
}
void L1RCTRegion::setNWEt(unsigned short nw){
  totalRegionEt.at(0) = nw;
}
void L1RCTRegion::setNWHE_FG(unsigned short nw){
  totalRegionHE_FG.at(0) = nw;
}

unsigned short L1RCTRegion::giveSWEt(){
  return getEtIn7Bits(0,3)&7;
}
unsigned short L1RCTRegion::giveSWHE_FG(){
  return getHE_FGBit(0,3);
}
void L1RCTRegion::setSWEt(unsigned short sw){
  totalRegionEt.at(30) = sw;
}
void L1RCTRegion::setSWHE_FG(unsigned short sw){
  totalRegionHE_FG.at(30) = sw;
}

unsigned short L1RCTRegion::giveSEEt(){
  return getEtIn7Bits(0,0)&7;
}
unsigned short L1RCTRegion::giveSEHE_FG(){
  return getHE_FGBit(0,0)&7;
}
void L1RCTRegion::setSEEt(unsigned short se){
  totalRegionEt.at(35) = se;
}
void L1RCTRegion::setSEHE_FG(unsigned short se){
  totalRegionHE_FG.at(35) = se;
}

void L1RCTRegion::print() {
  
  cout << " 7 Bit Energies ";
  for(int i = 0; i<4; i++){
    cout << endl;
    for(int j = 0; j<4; j++){
      cout << " " << getEtIn7Bits(i,j) << " ";
    }
  }

  cout << endl << endl;
  cout << " 9 Bit Energies ";
  for(int i = 0; i<4; i++){
    cout << endl;
    for(int j = 0; j<4; j++){
      cout << " " << getEtIn9Bits(i,j) << " ";
    }
  }
  
  cout << endl << endl;
  cout << " HE || FG bit ";
  for(int i = 0; i<4; i++){
    cout << endl;
    for(int j = 0; j<4; j++){
      cout << " " << getHE_FGBit(i,j) << " ";
    }
  }

  cout << endl << endl;
  cout << " Muon Bit ";
  for(int i = 0; i<4; i++){
    cout << endl;
    for(int j = 0; j<4; j++){
      cout << " " << getMuonBit(i,j) << " ";
    }
  }
  cout << endl;
}

void L1RCTRegion::printRaw(){
  for(int i = 0; i<16; i++){
    cout << totalRegionEt.at(i) << endl;
    cout << totalRegionHE_FG.at(i) << endl;
    cout << etIn9Bits.at(i) << endl;
    cout << muonBit.at(i) << endl;
  }
}

void L1RCTRegion::printEdges(){
  cout << "North" << endl;
  for(int i=0; i<4;i++)
    cout << totalRegionEt.at(i+1) << endl;
  
  cout << "West" << endl;
  for(int i=0; i<4;i++)
    cout << totalRegionEt.at(6*(i+1)) << endl;

  cout << "East" << endl;
  for(int i=0; i<4;i++)
    cout << totalRegionEt.at(6*(i+1)+5) << endl;
  
  cout << "South" << endl;
  for(int i=0; i<4;i++)
    cout << totalRegionEt.at(31+i) << endl;
 
  cout << "NE " << totalRegionEt.at(5) << endl;
  cout << "SE " << totalRegionEt.at(35) << endl;
  cout << "NW " << totalRegionEt.at(0) << endl;
  cout << "SW " << totalRegionEt.at(30) << endl;
}
