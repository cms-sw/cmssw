#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTORCAMap.h"
#include <iostream>
#include <vector>
using std::vector;
using std::cout;
using std::endl;
int main() {
  L1RCTORCAMap theMap;
  std::cout << theMap.combine(10,0) << " should equal 10" << std::endl;
  std::cout << theMap.combine(0,1) << " should equal 256" << std::endl;
  std::cout << theMap.combine(30,1) << " should equal 286" << std::endl;

  std::vector<bool> fgbool(56*72);
  std::vector<unsigned> het(56*72);
  std::vector<unsigned> hfet(18*8);
  std::vector<unsigned> eet(56*72);

  std::vector<unsigned short> et(56*72);
  et.at(0) = 1;
  std::vector<unsigned short> fg(56*72);
  fg.at(0) = 1;

  et.at(1) = 34;
  fg.at(1) = 0;

  std::vector<unsigned short> vec = theMap.combVec(et,fg);

  std::cout << vec.at(0) << " should equal 257" << std::endl;
  std::cout << vec.at(1) << " should equal 34" << std::endl;

  for(int phi=0;phi<72;phi++){
    for(int eta=0;eta<56;eta++){
      eet.at(56*phi+eta) = 56*phi+eta;
    }
  }
  theMap.readData(eet,het,fgbool,fgbool,hfet);
  std::vector<std::vector<std::vector<unsigned short> > > barrel = theMap.giveBarrel();
  
  for(int i=0;i<18;i++){
    for(int j=0;j<7;j++){
      for(int k=0;k<32;k++){
	cout << "crate " << i << " card " << j << " tower " << k << " value " << barrel.at(i).at(j).at(k) << std::endl;
      }
    }
  }

}
