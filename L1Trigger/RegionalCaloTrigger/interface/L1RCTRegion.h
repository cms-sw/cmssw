#ifndef L1RCTRegion_h
#define L1RCTRegion_h

#include <vector>

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
  
  //diagnostic print functions.
  //print prints the data contained in a convenient format
  //print edges prints the neighboring edge information
  void print();
  void printEdges();

  std::vector<unsigned short> giveNorthEt();
  std::vector<unsigned short> giveSouthEt();
  std::vector<unsigned short> giveWestEt();
  std::vector<unsigned short> giveEastEt();
  std::vector<unsigned short> giveNorthHE_FG();
  std::vector<unsigned short> giveSouthHE_FG();
  std::vector<unsigned short> giveWestHE_FG();
  std::vector<unsigned short> giveEastHE_FG();
  unsigned short giveSEEt();
  unsigned short giveSWEt();
  unsigned short giveNEEt();
  unsigned short giveNWEt();
  unsigned short giveSEHE_FG();
  unsigned short giveSWHE_FG();
  unsigned short giveNEHE_FG();
  unsigned short giveNWHE_FG();
  
  void setNorthEt(std::vector<unsigned short> north);
  void setSouthEt(std::vector<unsigned short> south);
  void setWestEt(std::vector<unsigned short> west);
  void setEastEt(std::vector<unsigned short> east);
  void setNorthHE_FG(std::vector<unsigned short> north);
  void setSouthHE_FG(std::vector<unsigned short> south);
  void setWestHE_FG(std::vector<unsigned short> west);
  void setEastHE_FG(std::vector<unsigned short> east);
  void setSEEt(unsigned short se);
  void setSWEt(unsigned short sw);
  void setNEEt(unsigned short ne);
  void setNWEt(unsigned short nw);
  void setSEHE_FG(unsigned short se);
  void setSWHE_FG(unsigned short sw);
  void setNEHE_FG(unsigned short ne);
  void setNWHE_FG(unsigned short nw);
  
 private:
  
  //6x6 matrices
  std::vector<unsigned short> totalRegionEt;
  std::vector<unsigned short> totalRegionHE_FG;
  //4x4 matrices
  std::vector<unsigned short> etIn9Bits;
  //std::vector<unsigned short> HE_FGBit;
  std::vector<unsigned short> muonBit;
  std::vector<unsigned short> activityBit;

};
#endif
