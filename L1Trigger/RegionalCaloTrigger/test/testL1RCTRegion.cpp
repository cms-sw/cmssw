#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRegion.h"

#include <vector>
using std::vector;
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

int main(){
  //Create a vector to be used in the borders and then set it to be
  //1 2 3 4 and use it as the borders to make sure that all the orientations
  //are being preserved correctly.  We'll also make sure all the corners
  //are working correctly.
  //We'll also set the Et of each of the elements in the main
  //region itself, 30-46
  //That way the region will ultimately look like *this*
  //7BitEt 10  1  2  3  4 11
  //       1  30 31 32 33  1
  //       2  34 35 36 37  2
  //       3  38 39 40 41  3
  //       4  42 43 44 45  4
  //       12  1  2  3  4 13

  //9BitEt    30 31 32 33
  //          34 35 36 37
  //          38 39 40 41
  //          42 43 44 45

  //HE_FG   1 0 1 0 1 0
  //        0 0 1 0 1 0
  //        1 0 1 0 1 1
  //        0 0 1 0 1 0
  //        1 0 1 0 1 1
  //        1 0 1 0 1 0

  std::vector<unsigned short> thing(4);
  std::vector<unsigned short> fgthing(4);
  fgthing.at(0)=0;
  fgthing.at(1)=1;
  fgthing.at(2)=0;
  fgthing.at(3)=1;
  thing.at(0)=1;
  thing.at(1)=2;
  thing.at(2)=3;
  thing.at(3)=4;
  unsigned short nw = 10, ne = 11, sw = 12, se = 13;
  unsigned short nwhe = 1, nehe = 0, swhe = 1, sehe = 0;
  L1RCTRegion region;
  for(int i = 0;i<4;i++){
    for(int j = 0;j<4;j++){
      region.setEtIn7Bits(i,j,30+j+4*i);
      region.setEtIn9Bits(i,j,30+j+4*i);
      region.setHE_FGBit(i,j,(4*i+j)%2);
    }
  }
  region.setNorthEt(thing);
  region.setNorthHE_FG(fgthing);
  region.setSouthEt(thing);
  region.setSouthHE_FG(fgthing);
  region.setWestEt(thing);
  region.setWestHE_FG(fgthing);
  region.setEastEt(thing);
  region.setEastHE_FG(fgthing);
  region.setSEEt(se);
  region.setSEHE_FG(sehe);
  region.setSWEt(sw);
  region.setSWHE_FG(swhe);
  region.setNEEt(ne);
  region.setNEHE_FG(nehe);
  region.setNWEt(nw);
  region.setNWHE_FG(nwhe);

  std::cout << "northEt ";
  for(int i=0;i<4;i++){
    std::cout << region.getEtIn7Bits(-1,i);
  }
  std::cout << std::endl;

  std::cout << "westEt ";
  for(int i=0;i<4;i++){
    std::cout << region.getEtIn7Bits(i,-1);
  }
  std::cout << std::endl;

  std::cout << "eastEt ";
  for(int i=0;i<4;i++){
    std::cout << region.getEtIn7Bits(i,4);
  }
  std::cout << std::endl;

  std::cout << "southEt ";
  for(int i=0;i<4;i++){
    std::cout << region.getEtIn7Bits(4,i);
  }
  std::cout << std::endl;

  std::cout << "given North Et ";
  std::vector<unsigned short> n = region.giveNorthEt();
  for(int i=0;i<4;i++){
    std::cout << n.at(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "given North HEFG ";
  std::vector<unsigned short> nhe = region.giveNorthHE_FG();
  for(int i=0;i<4;i++){
    std::cout << nhe.at(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "given South Et ";
  std::vector<unsigned short> s = region.giveSouthEt();
  for(int i=0;i<4;i++){
    std::cout << s.at(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "given South HEFG ";
  std::vector<unsigned short> she = region.giveSouthHE_FG();
  for(int i=0;i<4;i++){
    std::cout << she.at(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "given East et ";
  std::vector<unsigned short> e = region.giveEastEt();
  for(int i=0;i<4;i++){
    std::cout << e.at(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "given East HEFG ";
  std::vector<unsigned short> ehe = region.giveEastEt();
  for(int i=0;i<4;i++){
    std::cout << e.at(i) << " ";
  }
  std::cout << std::endl;

  std::cout << "given West et ";
  std::vector<unsigned short> w = region.giveWestEt();
  for(int i=0;i<4;i++){
    std::cout << w.at(i) << " ";
  }
  std::cout << std::endl;


  std::cout << "7BitEt ";
  for(int i=-1;i<5;i++){
    std::cout << std::endl;
    for(int j=-1;j<5;j++){
      std::cout << region.getEtIn7Bits(i,j) << " ";
    }
  }
  std::cout << std::endl;
  std::cout << "HEFG ";
  for(int i=-1;i<5;i++){
    std::cout << std::endl;
    for(int j=-1;j<5;j++){
      std::cout << region.getHE_FGBit(i,j) << " ";
    }
  }
  std::cout << std::endl;
}
  
  
