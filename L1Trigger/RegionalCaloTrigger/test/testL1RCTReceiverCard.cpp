#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTReceiverCard.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include <vector>
using std::vector;
int main() {
  std::string filename("../data/TPGcalc.txt");
  L1RCTLookupTables lut(filename);
  L1RCTReceiverCard flip(9,0);
  L1RCTReceiverCard card(0,0);
  L1RCTReceiverCard six(0,6);
  vector<unsigned short> input1(64);
  vector<unsigned short> input2(64);
  vector<unsigned short> input3(64);
  vector<unsigned short> input4(64);
  input1.at(0) = 100;
  input1.at(1) = 100;
  input1.at(7) = 100;
  //All these inputs should go into the ecal.
  //They should be at positions 
  //0  0  0    100
  //0  0  0    100
  //0  0  0    0
  //0  0  100  0
  //The energy sum should be 300 and
  //the tau bit should be set to on because
  //the phi pattern is 1101
  card.fillInput(input1, &lut);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  card.print();
  

  //The following should look like
  //0 0 0 100
  //0 0 0 0
  //0 0 0 0
  //0 0 0 0
  //The tau bit should be off.
  input2.at(0) = 50;
  input2.at(32) = 50;
  card.fillInput(input2, &lut);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  card.print();
  
  //The following should look like
  //0 0 0 100
  //0 0 0 100
  //0 0 0 0
  //0 0 0 0
  //and have the muon bit on at tower 0
  //and have the fg bit on at tower 1 and the he bit on at tower 0
  //because the h is greater than the e.  The muon bit for region 0
  //should be on and the tau bit should be off.
  input3.at(32) = 356;
  input3.at(1) = 356;
  card.fillInput(input3,&lut);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  card.print();
  
  //Let's make sure that everything can be set correctly in all the regions
  //We'll set the energies to be the same as the tower number+1 and
  //make sure it matches with the layout of the receiver card that
  //we want, kthnx.

  for(int i =0;i<32;i++)
    input4.at(i)=i+1;
  card.fillInput(input4,&lut);
  card.fillMuonBits();
  card.fillTauBits();
  card.fillRegionSums();
  card.print();

  flip.fillInput(input4,&lut);
  flip.fillMuonBits();
  flip.fillTauBits();
  flip.fillRegionSums();
  flip.print();

  six.fillInput(input4,&lut);
  six.fillMuonBits();
  six.fillTauBits();
  six.fillRegionSums();
  six.print();

}
