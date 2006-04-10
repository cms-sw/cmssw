#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTReceiverCard.h"
#include <vector>
using std::vector;
int main() {
  L1RCTReceiverCard card(0,0);
  vector<unsigned short> input1(64);
  vector<unsigned short> input2(64);
  vector<unsigned short> input3(64);
  input1.at(0) = 100;
  input1.at(1) = 100;
  input1.at(7) = 100;
  card.fillInput(input1);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  card.print();
  
  input2.at(0) = 50;
  input2.at(32) = 50;
  card.fillInput(input2);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  //card.print();

  input3.at(32) = 356;
  input3.at(1) = 356;
  card.fillInput(input3);
  card.fillTauBits();
  card.fillRegionSums();
  card.fillMuonBits();
  //card.print();
}
