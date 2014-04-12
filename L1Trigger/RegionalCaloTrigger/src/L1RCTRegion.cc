#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRegion.h"

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

L1RCTRegion::L1RCTRegion() : totalRegionEt(36),
			     totalRegionHE_FG(36),
			     etIn9Bits(16),
			     muonBit(16),
			     activityBit(16)
{
}


L1RCTRegion::~L1RCTRegion()
{}

//So the whole point of the following two functions is that they provide
//an interface to the "real" 4x4 region 7 bit energies and h/e||fg bits
//that are used in electron finding.  
//Now, you actually *can* give them arguments ranging from -1 to 4 
//representing the outer neighbors.
//This is actually quite helpful and allows you to write the electronfinding
//algorithm in the same form no matter what tower you're centered on.
//
//As a reminder i is row and j is column, just like matrices.
//row -1 is the northern neighbors and column -1 is the western neighbors
unsigned short L1RCTRegion::getEtIn7Bits(int i, int j) const{
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


unsigned short L1RCTRegion::getHE_FGBit(int i, int j) const{
  return totalRegionHE_FG.at(6*(i+1)+j+1);
}

void L1RCTRegion::setHE_FGBit(int i, int j, unsigned short HE_FG){
  totalRegionHE_FG.at(6*(i+1)+j+1) = HE_FG;
}


//The rest of the data stored in a region only works if i and j are
//in the 0-3 range.  The arrays truly are 4x4 and will signal an error
//if misused thanks to the vector function .at
unsigned short L1RCTRegion::getEtIn9Bits(int i, int j) const{
  return etIn9Bits.at(4*i + j);
}

void L1RCTRegion::setEtIn9Bits(int i, int j,unsigned short energy){
  if(energy <=511)
    etIn9Bits.at(4*i+j) = energy;
  else 
    etIn9Bits.at(4*i+j) = 511;
}

unsigned short L1RCTRegion::getMuonBit(int i, int j) const{
  return muonBit.at(4*i+j);
}

void L1RCTRegion::setMuonBit(int i, int j,unsigned short muon) {
  muonBit.at(4*i+j) = muon;
}

void L1RCTRegion::setActivityBit(int i, int j, unsigned short activity){
  activityBit.at(4*i+j) = activity;
}

unsigned short L1RCTRegion::getActivityBit(int i, int j) const{
  return activityBit.at(4*i+j);
}

//The following list of give and set functions are the core
//of the work for neighbor sharing swept under the rug.
//Basically, the way it works is that "give" methods return
//what would be the appropriate neighbor information so that you can
//use the set methods on the other region in order to set the neighbor
//information.  For example, r0:crate 0 card 0 region 0 is the northern
//neighbor of r1:crate 0 card 1 region 0.  Then to set the northern
//neighbor information you call r1.setNorthEt(r0.getNorthEt())
//That's why it's give insted of get.  It doesn't return the region's
//northern neighbor information, it returns what would be its southern
//neighbor's northern neighbor information.
vector<unsigned short> L1RCTRegion::giveNorthEt() const{
  std::vector<unsigned short> north(4);
  for(int i = 0; i<4;i++)
    north.at(i) = getEtIn7Bits(3,i);
  return north;
}
void L1RCTRegion::setNorthEt(const std::vector<unsigned short>& north) {
  for(int i = 0; i<4; i++)
    totalRegionEt.at(i+1) = north.at(i);
}
vector<unsigned short> L1RCTRegion::giveNorthHE_FG() const{
  std::vector<unsigned short> north(4);
  for(int i = 0; i<4; i++)
    north.at(i) = getHE_FGBit(3,i);
  return north;
}
void L1RCTRegion::setNorthHE_FG(const std::vector<unsigned short>& north){
  for(int i = 0; i<4; i++)
    totalRegionHE_FG.at(i+1) = north.at(i);
}

vector<unsigned short> L1RCTRegion::giveSouthEt() const{
  std::vector<unsigned short> south(4);
  for(int i = 0; i<4; i++)
    south.at(i) = getEtIn7Bits(0,i);
  return south;
}
void L1RCTRegion::setSouthEt(const std::vector<unsigned short>& south){
  for(int i = 0; i<4; i++)
    totalRegionEt.at(31+i) = south.at(i);
}

vector<unsigned short> L1RCTRegion::giveSouthHE_FG() const{
  std::vector<unsigned short> south(4);
  for(int i = 0; i<4; i++)
    south.at(i) = getHE_FGBit(0,i);
  return south;
}
void L1RCTRegion::setSouthHE_FG(const std::vector<unsigned short>& south){
  for(int i=0; i<4; i++)
    totalRegionHE_FG.at(31+i) = south.at(i);
}

vector<unsigned short> L1RCTRegion::giveWestEt() const{
  std::vector<unsigned short> west(4);
  for(int i =0; i<4; i++)
    west.at(i) = getEtIn7Bits(i,3);
  return west;
}
void L1RCTRegion::setWestEt(const std::vector<unsigned short>& west){
  for(int i = 0; i<4; i++)
    totalRegionEt.at(6*(i+1)) = west.at(i);
}

vector<unsigned short> L1RCTRegion::giveWestHE_FG() const{
  std::vector<unsigned short> west(4);
  for(int i = 0; i<4; i++)
    west.at(i) = getHE_FGBit(i,3);
  return west;
}
void L1RCTRegion::setWestHE_FG(const std::vector<unsigned short>& west){
  for(int i = 0; i<4; i++)
    totalRegionHE_FG.at(6*(i+1)) = west.at(i);
}

vector<unsigned short> L1RCTRegion::giveEastEt() const{
  std::vector<unsigned short> east(4);
  for(int i = 0; i<4; i++)
    east.at(i) = getEtIn7Bits(i,0);
  return east;
}
void L1RCTRegion::setEastEt(const std::vector<unsigned short>& east){
  for(int i = 0; i<4; i++)
    totalRegionEt.at(6*(i+1) + 5) = east.at(i);
}

vector<unsigned short> L1RCTRegion::giveEastHE_FG() const{
  std::vector<unsigned short> east(4);
  for(int i = 0; i<4; i++)
    east.at(i) = getHE_FGBit(i,0);
  return east;
}
void L1RCTRegion::setEastHE_FG(const std::vector<unsigned short>& east){
  for(int i = 0; i<4; i++)
    totalRegionHE_FG.at(6*(i+1) + 5) = east.at(i);
}

unsigned short L1RCTRegion::giveNEEt() const{
  unsigned short et = getEtIn7Bits(3,0);
  if(et > 7)
    return 7;
  else
    return et;
}
unsigned short L1RCTRegion::giveNEHE_FG() const{
  return getHE_FGBit(3,0);
}
void L1RCTRegion::setNEEt(unsigned short ne){
  totalRegionEt.at(5) = ne;
}
void L1RCTRegion::setNEHE_FG(unsigned short ne){
  totalRegionHE_FG.at(5) = ne;
}

unsigned short L1RCTRegion::giveNWEt() const{
  unsigned short et = getEtIn7Bits(3,3);
  if(et > 7)
    return 7;
  else 
    return et;
}
unsigned short L1RCTRegion::giveNWHE_FG() const{
  return getHE_FGBit(3,3);
}
void L1RCTRegion::setNWEt(unsigned short nw){
  totalRegionEt.at(0) = nw;
}
void L1RCTRegion::setNWHE_FG(unsigned short nw){
  totalRegionHE_FG.at(0) = nw;
}

unsigned short L1RCTRegion::giveSWEt() const{
  unsigned short et = getEtIn7Bits(0,3);
  if(et > 7)
    return 7;
  else
    return et;
}
unsigned short L1RCTRegion::giveSWHE_FG() const{
  return getHE_FGBit(0,3);
}
void L1RCTRegion::setSWEt(unsigned short sw){
  totalRegionEt.at(30) = sw;
}
void L1RCTRegion::setSWHE_FG(unsigned short sw){
  totalRegionHE_FG.at(30) = sw;
}

unsigned short L1RCTRegion::giveSEEt() const{
  unsigned short et = getEtIn7Bits(0,0);
  if(et > 7)
    return 7;
  else
    return et;
}
unsigned short L1RCTRegion::giveSEHE_FG() const{
  return getHE_FGBit(0,0);
}
void L1RCTRegion::setSEEt(unsigned short se) {
  totalRegionEt.at(35) = se;
}
void L1RCTRegion::setSEHE_FG(unsigned short se) {
  totalRegionHE_FG.at(35) = se;
}

void L1RCTRegion::print() {
  
  std::cout << " 7 Bit Energies ";
  for(int i = 0; i<4; i++){
    std::cout << std::endl;
    for(int j = 0; j<4; j++){
      std::cout << " " << getEtIn7Bits(i,j) << " ";
    }
  }

  std::cout << std::endl << std::endl;
  std::cout << " 9 Bit Energies ";
  for(int i = 0; i<4; i++){
    std::cout << std::endl;
    for(int j = 0; j<4; j++){
      std::cout << " " << getEtIn9Bits(i,j) << " ";
    }
  }
  
  std::cout << std::endl << std::endl;
  std::cout << " HE || FG bit ";
  for(int i = 0; i<4; i++){
    std::cout << std::endl;
    for(int j = 0; j<4; j++){
      std::cout << " " << getHE_FGBit(i,j) << " ";
    }
  }

  std::cout << std::endl << std::endl;
  std::cout << " Muon Bit ";
  for(int i = 0; i<4; i++){
    std::cout << std::endl;
    for(int j = 0; j<4; j++){
      std::cout << " " << getMuonBit(i,j) << " ";
    }
  }
  std::cout << std::endl;
}

void L1RCTRegion::printEdges(){
  std::cout << "North" << std::endl;
  for(int i=0; i<4;i++)
    std::cout << totalRegionEt.at(i+1) << std::endl;
  
  std::cout << "West" << std::endl;
  for(int i=0; i<4;i++)
    std::cout << totalRegionEt.at(6*(i+1)) << std::endl;

  std::cout << "East" << std::endl;
  for(int i=0; i<4;i++)
    std::cout << totalRegionEt.at(6*(i+1)+5) << std::endl;
  
  std::cout << "South" << std::endl;
  for(int i=0; i<4;i++)
    std::cout << totalRegionEt.at(31+i) << std::endl;
 
  std::cout << "NE " << totalRegionEt.at(5) << std::endl;
  std::cout << "SE " << totalRegionEt.at(35) << std::endl;
  std::cout << "NW " << totalRegionEt.at(0) << std::endl;
  std::cout << "SW " << totalRegionEt.at(30) << std::endl;
}
