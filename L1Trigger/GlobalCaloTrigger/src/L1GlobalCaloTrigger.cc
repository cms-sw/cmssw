#include "../interface/L1GlobalCaloTrigger.h"

L1GlobalCaloTrigger* L1GlobalCaloTrigger::instance = 0;

// constructor - create the modules
L1GlobalCaloTrigger::L1GlobalCaloTrigger() {
	
	caloConcCard = new L1GctCaloConcentratorCard();
	
	
	muonConcCard = new L1GctMuonConcentratorCard();
	
	plusWheelCard = new L1GctWheelCard();
	for (int i=0; i<3; i++) {
		L1GctJetLeafCard* lc = new L1GctJetLeafCard();
		plusWheelCard->addLeafCard(lc);
	}
		
	minusWheelCard = new L1GctWheelCard();
	
	for (int i=0; i<18; i++) {
		L1GctSourceCard* sc = new L1GctSourceCard();
		theSourceCards.push_back(sc);
	}
	
}

L1GlobalCaloTrigger::~L1GlobalCaloTrigger()
{
	delete caloConcCard;
	delete muonConcCard;

	delete plusWheelCard;
	delete minusWheelCard;
	
	theSourceCards.clear();
}

L1GlobalCaloTrigger* L1GlobalCaloTrigger::theGct() {
	if (L1GlobalCaloTrigger::instance==0) {
		L1GlobalCaloTrigger::instance = new L1GlobalCaloTrigger();
	}
	return L1GlobalCaloTrigger::instance;
}

void L1GlobalCaloTrigger::process() {
	
	
	
}