#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include <vector>
#include <iostream>
using std::vector;
using std::cout;
using std::endl;

vector<vector<vector<unsigned short> > > zeroVec(){
  vector<vector<vector<unsigned short> > > v(18,vector<vector<unsigned short> >(7,
				           vector<unsigned short>(64)));
  return v;
} 

int main(){
  std::string filename("../data/TPGcalc.txt");
  L1RCT rct(filename);
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,
					    vector<unsigned short>(64)));
  for(int j = 0; j<4; j++){
    for(int i = 0; i<8; i++){
      barrel = zeroVec();
      barrel.at(0).at(0).at(j+4*i) = 10;
      rct.input(barrel,hf);
      rct.processEvent();
      //rct.printEIC(0,0);
      
      barrel = zeroVec();
      barrel.at(0).at(0).at(j+4*i) = 5;
      if(i<7)
	barrel.at(0).at(0).at(j+4*(i+1)) = 5;
      else
	barrel.at(0).at(2).at(j) = 5;
      rct.input(barrel,hf);
      rct.processEvent();
      rct.printEIC(0,0);
      //rct.printEICEdges(0,0);
      rct.printEIC(0,2);
      
    }
  }
  
}
