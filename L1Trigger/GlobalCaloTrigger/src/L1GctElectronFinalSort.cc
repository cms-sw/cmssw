#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

L1GctElectronFinalSort::L1GctElectronFinalSort(bool iso, L1GctEmLeafCard* posEtaCard,
                                                         L1GctEmLeafCard* negEtaCard):
  L1GctProcessor(),
  m_emCandsType(iso),
  m_thePosEtaLeafCard(nullptr), m_theNegEtaLeafCard(nullptr),
  m_inputCands(16),
  m_outputCands(4),
  m_setupOk(true)
{
  if(posEtaCard!=nullptr){
    m_thePosEtaLeafCard = posEtaCard;
  }else{
    m_setupOk = false;
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	<<"L1GctElectronFinalSort::Constructor() : 1st EmLeafCard passed is zero";
    }
  }
  if(negEtaCard!=nullptr){
    m_theNegEtaLeafCard = negEtaCard;
  }else{
    m_setupOk = false;
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	<<"L1GctElectronFinalSort::Constructor() : 2nd EmLeafCard passed is zero";
    }
  }
  if (!m_setupOk && m_verbose) {
    edm::LogError("L1GctSetupError") << "L1GctElectronFinalSort has been incorrectly constructed";
  }
}

L1GctElectronFinalSort::~L1GctElectronFinalSort(){
  m_inputCands.clear();
  m_outputCands.contents.clear();
}

void L1GctElectronFinalSort::resetProcessor() {
  m_inputCands.clear();
  m_inputCands.resize(16);
}

void L1GctElectronFinalSort::resetPipelines() {
  m_outputCands.reset(numOfBx());
}

void L1GctElectronFinalSort::fetchInput() {
  if (m_setupOk) {
    for (int k=0; k<4; k++) {  /// loop over candidates from four electron sorter FPGAs
      if (m_emCandsType) {
	setInputEmCand(  k  , m_thePosEtaLeafCard->getIsoElectronSorterU1()->getOutputCands().at(k)); 
	setInputEmCand( k+4 , m_thePosEtaLeafCard->getIsoElectronSorterU2()->getOutputCands().at(k)); 
	setInputEmCand( k+8 , m_theNegEtaLeafCard->getIsoElectronSorterU1()->getOutputCands().at(k)); 
	setInputEmCand( k+12, m_theNegEtaLeafCard->getIsoElectronSorterU2()->getOutputCands().at(k)); 
      }
      else {
	setInputEmCand(  k  , m_thePosEtaLeafCard->getNonIsoElectronSorterU1()->getOutputCands().at(k)); 
	setInputEmCand( k+4 , m_thePosEtaLeafCard->getNonIsoElectronSorterU2()->getOutputCands().at(k)); 
	setInputEmCand( k+8 , m_theNegEtaLeafCard->getNonIsoElectronSorterU1()->getOutputCands().at(k)); 
	setInputEmCand( k+12, m_theNegEtaLeafCard->getNonIsoElectronSorterU2()->getOutputCands().at(k)); 
      }
    }
  }
}

void L1GctElectronFinalSort::process(){

  if (m_setupOk) {
    std::vector<prioritisedEmCand> data(m_inputCands.size());
    // Assign a "priority" for sorting - this assumes the candidates
    // have already been filled in "priority order"
    for (unsigned i=0; i<m_inputCands.size(); i++) {
      prioritisedEmCand c(m_inputCands.at(i), i);
      data.at(i) = c;
    }

    //Then sort it
    sort(data.begin(),data.end(),L1GctElectronSorter::rankByGt);
  
    //Copy data to output buffer
    std::vector<L1GctEmCand> temp(4);
    for(int i = 0; i<4; i++){
      temp.at(i) = data.at(i).emCand;
    }
    m_outputCands.store(temp, bxRel());
  }
}

void L1GctElectronFinalSort::setInputEmCand(unsigned i, const L1GctEmCand& cand){
  if ((i<m_inputCands.size()) 
    && (cand.bx()==bxAbs())) {
    m_inputCands.at(i) = cand;
  }
}

std::ostream& operator<<(std::ostream& s, const L1GctElectronFinalSort& cand) {
  s << "===ElectronFinalSort===" << std::endl;
  s << "Card type = " << ( cand.m_emCandsType ? "isolated" : "non-isolated" ) <<std::endl;
  s << "Pointers to the Electron Leaf cards are: "<<std::endl;
  s << "   Pos. eta: " << cand.m_thePosEtaLeafCard;
  s << "   Neg. eta: " << cand.m_theNegEtaLeafCard;
  s << std::endl;
  s << "No of Electron Input Candidates " << cand.m_inputCands.size() << std::endl;
  s << "No of Electron Output Candidates " << cand.m_outputCands.contents.size() << std::endl;
   
  return s;
}


