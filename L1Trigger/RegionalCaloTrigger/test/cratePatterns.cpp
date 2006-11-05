#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTORCAMap.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
using std::vector;
using std::fstream;
using std::cout;
using std::endl;
using std::ios;

int main (){
  L1RCTORCAMap theMap;
  std::string filename("../data/TPGcalc.txt");
  L1RCT rct(filename);
  L1RCTLookupTables* lut = rct.getLUT();
  vector<int> data(4);
  vector<int> location(3);
  unsigned long lookupValue;
  vector<vector<unsigned short> > hf(18,vector<unsigned short>(8));
  vector<vector<vector<unsigned short> > > barrel(18,vector<vector<unsigned short> >(7,
                                            vector<unsigned short>(64)));

  char throwaway[2000];
  //Now we pull in the data from the crate.input file
  fstream input("crate.input",ios::in);
  fstream output("lut.out",ios::out);
  fstream rctoutput("rct.out",ios::out);
  input.getline(throwaway,2000);
  output << "Eta    Phi    Ecal    Hcal    Lookup" << endl;
  while(!input.eof()){
    for(int i=0;i<4;i++){
      input >> data.at(i);
      output << data.at(i) << "      ";
    }
      location = theMap.orcamap(data.at(0),data.at(1));
      barrel.at(location.at(0)).at(location.at(1)).at(location.at(2)) = data.at(2);
      barrel.at(location.at(0)).at(location.at(1)).at(location.at(2)+32) = data.at(3);
      lookupValue = lut->lookup(data.at(2)&255,data.at(3)&255,(data.at(2)<<8)&1,
				location.at(0), location.at(1), location.at(2));
      output << lookupValue << endl;
  }
  rct.input(barrel,hf);
  rct.processEvent();
  input.close();
  output.close();
  rct.printCrate(0);
}
