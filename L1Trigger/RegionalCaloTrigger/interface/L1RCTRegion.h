#ifndef L1RCTRegion_h
#define L1RCTRegion_h

#include <vector>

class L1RCTRegion {

 public:

  L1RCTRegion();
  ~L1RCTRegion();
 
  void setEtIn7Bits(int i, int j,unsigned short energy);
  unsigned short getEtIn9Bits(int i, int j) const;
  void setEtIn9Bits(int i, int j,unsigned short energy);
  void setHE_FGBit(int i, int j,unsigned short HE_FG);
  unsigned short getMuonBit(int i, int j) const;
  void setMuonBit(int i, int j, unsigned short muon);
  void setActivityBit(int i, int j, unsigned short activity);
  unsigned short getActivityBit(int i, int j) const;

  unsigned short getEtIn7Bits(int i, int j) const;
  unsigned short getHE_FGBit(int i, int j) const;
  
  //diagnostic print functions.
  //print prints the data contained in a convenient format
  //print edges prints the neighboring edge information
  void print();
  void printEdges();

  std::vector<unsigned short> giveNorthEt() const;
  std::vector<unsigned short> giveSouthEt() const;
  std::vector<unsigned short> giveWestEt() const;
  std::vector<unsigned short> giveEastEt() const;
  std::vector<unsigned short> giveNorthHE_FG() const;
  std::vector<unsigned short> giveSouthHE_FG() const;
  std::vector<unsigned short> giveWestHE_FG() const;
  std::vector<unsigned short> giveEastHE_FG() const;
  unsigned short giveSEEt() const;
  unsigned short giveSWEt() const;
  unsigned short giveNEEt() const;
  unsigned short giveNWEt() const;
  unsigned short giveSEHE_FG() const;
  unsigned short giveSWHE_FG() const;
  unsigned short giveNEHE_FG() const;
  unsigned short giveNWHE_FG() const;
  
  void setNorthEt(const std::vector<unsigned short>& north);
  void setSouthEt(const std::vector<unsigned short>& south);
  void setWestEt(const std::vector<unsigned short>& west);
  void setEastEt(const std::vector<unsigned short>& east);
  void setNorthHE_FG(const std::vector<unsigned short>& north);
  void setSouthHE_FG(const std::vector<unsigned short>& south);
  void setWestHE_FG(const std::vector<unsigned short>& west);
  void setEastHE_FG(const std::vector<unsigned short>& east);
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
