#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelCard.h"

L1GctWheelCard::L1GctWheelCard() {
		
	for (int i=0; i<3; i++) {
		L1GctJetLeafCard* lc = new L1GctJetLeafCard();
		jetLeafCards.push_back(lc);
	}
		
}

L1GctWheelCard::~L1GctWheelCard() {

	for (int i=0; i<3; i++) {
//		L1GctJetLeafCard* lc = jetLeafCards.pop_front();
//		delete lc;
	}
	
}
