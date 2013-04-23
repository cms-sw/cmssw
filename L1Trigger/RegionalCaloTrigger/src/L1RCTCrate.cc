#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCrate.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"

L1RCTCrate::L1RCTCrate(int crtNo, const L1RCTLookupTables* rctLookupTables) : 
  jetSummaryCard(crtNo, rctLookupTables),
  crtNo(crtNo),
  rctLookupTables_(rctLookupTables)
{
  for(int i = 0; i <7; i++){
    L1RCTReceiverCard rc(crtNo,i,rctLookupTables);
    L1RCTElectronIsolationCard eic(crtNo,i,rctLookupTables);
    receiverCards.push_back(rc);
    electronCards.push_back(eic);
  }
}

L1RCTCrate::~L1RCTCrate()
{

}

void L1RCTCrate::input(const std::vector<std::vector<unsigned short> >& RCInput,
		       const std::vector<unsigned short>& HFInput)
{
  //std::cout << "Crate.input() entered" << std::endl;
  for(int i =0; i<7; i++){
    //std::cout << "calling RC.fillInput() for RC " << i << std::endl;
    receiverCards.at(i).fillInput(RCInput.at(i));
    //std::cout << "RC " << i << " filled" << std::endl;
  }
  //std::cout << "calling JSC.fillHFRegionSums()" << std::endl;
  jetSummaryCard.fillHFRegionSums(HFInput);
  //std::cout << "JSC.fillHF called" << std::endl;
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
      electronCards.at(i).setRegion(j,*receiverCards.at(i).getRegion(j));
    }
  }
}
void L1RCTCrate::processElectronIsolationCards(){
  for(int i = 0; i<7;i++)
    electronCards.at(i).fillElectronCandidates();
}
void L1RCTCrate::fillJetSummaryCard(){
  std::vector<unsigned short> barrelSums(14);
  std::vector<unsigned short> isoElectrons(14);
  std::vector<unsigned short> nonIsoElectrons(14);
  std::vector<unsigned short> mipBits(14);
  std::vector<unsigned short> overFlowBits(14);
  std::vector<unsigned short> tauBits(14);
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
