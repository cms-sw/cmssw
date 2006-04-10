#ifndef L1RCTRegion_h
#define L1RCTRegion_h

#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;


class L1RCTRegion {

 public:

  L1RCTRegion();
 
  void setEtIn7Bits(int i, int j,unsigned short energy);
  unsigned short getEtIn9Bits(int i, int j);
  void setEtIn9Bits(int i, int j,unsigned short energy);
  void setHE_FGBit(int i, int j,unsigned short HE_FG);
  unsigned short getMuonBit(int i, int j);
  void setMuonBit(int i, int j, unsigned short muon);
  void setActivityBit(int i, int j, unsigned short activity);
  unsigned short getActivityBit(int i, int j);

  unsigned short getEtIn7Bits(int i, int j);
  unsigned short getHE_FGBit(int i, int j);

  void print();
  void printRaw();
  void printEdges();

  vector<unsigned short> giveNorthEt();
  vector<unsigned short> giveSouthEt();
  vector<unsigned short> giveWestEt();
  vector<unsigned short> giveEastEt();
  vector<unsigned short> giveNorthHE_FG();
  vector<unsigned short> giveSouthHE_FG();
  vector<unsigned short> giveWestHE_FG();
  vector<unsigned short> giveEastHE_FG();
  unsigned short giveSEEt();
  unsigned short giveSWEt();
  unsigned short giveNEEt();
  unsigned short giveNWEt();
  unsigned short giveSEHE_FG();
  unsigned short giveSWHE_FG();
  unsigned short giveNEHE_FG();
  unsigned short giveNWHE_FG();
  
  void setNorthEt(vector<unsigned short> north);
  void setSouthEt(vector<unsigned short> south);
  void setWestEt(vector<unsigned short> west);
  void setEastEt(vector<unsigned short> east);
  void setNorthHE_FG(vector<unsigned short> north);
  void setSouthHE_FG(vector<unsigned short> south);
  void setWestHE_FG(vector<unsigned short> west);
  void setEastHE_FG(vector<unsigned short> east);
  void setSEEt(unsigned short se);
  void setSWEt(unsigned short sw);
  void setNEEt(unsigned short ne);
  void setNWEt(unsigned short nw);
  void setSEHE_FG(unsigned short se);
  void setSWHE_FG(unsigned short sw);
  void setNEHE_FG(unsigned short ne);
  void setNWHE_FG(unsigned short nw);
  
 private:
  
  //4x4 array
  vector<unsigned short> totalRegionEt;
  vector<unsigned short> totalRegionHE_FG;
  vector<unsigned short> etIn9Bits;
  //vector<unsigned short> HE_FGBit;
  vector<unsigned short> muonBit;
  vector<unsigned short> activityBit;

};
#endif
