#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include <time.h>
#include <iostream>
using namespace std;

int main (){
  double diff;
  std::string filename("../data/TPGcalc.txt");
  L1RCT l1rct(filename);
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,vector<unsigned short>(64)));
  time_t s1 = time (NULL);
  for(int i=0;i<10000;i++){
    l1rct.input(barrel,hf);
    l1rct.processEvent();
  }
  time_t s2 = time (NULL);
  diff =difftime (s2,s1);
  cout << diff << endl;
}
