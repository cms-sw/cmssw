#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTORCAMap.h"
#include <iostream>
using std::cout;
using std::endl;
int main(){
  L1RCTORCAMap themap;
  for(int i = 0; i<72; i++){
    for(int j = 0; j<56; j++){
      vector<int> vec = themap.orcamap(j,i);
      cout << "phi " << i << " eta " <<  j << " goes to ";
      for(int k = 0; k < 3; k++)
	cout << vec.at(k) << " ";
      cout << endl;
    }
  }
}
