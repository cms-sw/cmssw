#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCrate.h"

L1RCTCrate::L1RCTCrate(int crtNo) : jetSummaryCard(crtNo),crtNo(crtNo)
{
  for(int i = 0; i <7; i++){
    L1RCTReceiverCard rc(crtNo,i);
    L1RCTElectronIsolationCard eic(crtNo,i);
    receiverCards.push_back(rc);
    electronCards.push_back(eic);
  }
}
void L1RCTCrate::input(vector<vector<unsigned short> > RCInput,
		       vector<unsigned short> HFInput){
  //cout << "Crate.input() entered" << endl;
  for(int i =0; i<7; i++){
    //cout << "calling RC.fillInput() for RC " << i << endl;
    receiverCards.at(i).fillInput(RCInput.at(i));
    //cout << "RC " << i << " filled" << endl;
  }
  //cout << "calling JSC.fillHFRegionSums()" << endl;
  jetSummaryCard.fillHFRegionSums(HFInput);
  //cout << "JSC.fillHF called" << endl;
} 
void L1RCTCrate::processReceiverCards(){
  for(int i=0; i<7;i++){
    receiverCards.at(i).fillTauBits();
    receiverCards.at(i).fillRegionSums();
    receiverCards.at(i).fillMuonBits();
  }
}
void L1RCTCrate::fillElectronIsolationCards(){
  for(int i = 0; i<7; i++){
    for(int j = 0; j<2; j++){
      electronCards.at(i).setRegion(j,receiverCards.at(i).getRegion(j));
    }
  }
}
void L1RCTCrate::processElectronIsolationCards(){
  for(int i = 0; i<7;i++)
    electronCards.at(i).fillElectronCandidates();
}
void L1RCTCrate::fillJetSummaryCard(){
  vector<unsigned short> barrelSums(14);
  vector<unsigned short> isoElectrons(14);
  vector<unsigned short> nonIsoElectrons(14);
  vector<unsigned short> mipBits(14);
  vector<unsigned short> overFlowBits(14);
  vector<unsigned short> tauBits(14);
  for(int i = 0; i<7;i++){
    mipBits.at(2*i) = receiverCards.at(i).getMuonBitRegion(0);
    mipBits.at(2*i+1) = receiverCards.at(i).getMuonBitRegion(1);
    isoElectrons.at(2*i) = electronCards.at(i).getIsoElectrons(0);
    isoElectrons.at(2*i+1) = electronCards.at(i).getIsoElectrons(1) + 1;  // the +1 adds region info
    nonIsoElectrons.at(2*i) = electronCards.at(i).getNonIsoElectrons(0);
    nonIsoElectrons.at(2*i+1) = electronCards.at(i).getNonIsoElectrons(1) + 1;  // +1 adds region info
    barrelSums.at(2*i) = receiverCards.at(i).getEtIn10BitsRegion(0);
    barrelSums.at(2*i+1) = receiverCards.at(i).getEtIn10BitsRegion(1);
    overFlowBits.at(2*i) = receiverCards.at(i).getOverFlowBitRegion(0);
    overFlowBits.at(2*i+1) = receiverCards.at(i).getOverFlowBitRegion(1);
    tauBits.at(2*i) = receiverCards.at(i).getTauBitRegion(0);
    tauBits.at(2*i+1) = receiverCards.at(i).getTauBitRegion(1);
  }
  jetSummaryCard.fillIsolatedEGObjects(isoElectrons);
  jetSummaryCard.fillNonIsolatedEGObjects(nonIsoElectrons);
  jetSummaryCard.fillRegionSums(barrelSums);
  jetSummaryCard.fillMIPBits(mipBits);
  jetSummaryCard.fillTauBits(tauBits);
  jetSummaryCard.fillOverFlowBits(overFlowBits);
}
void L1RCTCrate::processJetSummaryCard(){
  jetSummaryCard.fillJetRegions();
  jetSummaryCard.fillQuietBits();
}

void L1RCTCrate::print(){
  for(int i=0;i<7;i++){
    receiverCards.at(i).print();
    electronCards.at(i).print();
  }
  jetSummaryCard.print();
}
