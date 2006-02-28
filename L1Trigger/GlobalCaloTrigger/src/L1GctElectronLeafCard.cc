#include "../interface/L1GctElectronLeafCard.h"

L1GctElectronLeafCard::L1GctElectronLeafCard() {
}

L1GctElectronLeafCard::~L1GctElectronLeafCard() {
}

void L1GctElectronLeafCard::process() {
	
	
	
}

void L1GctElectronLeafCard::addSource(L1GctSourceCard* card) {
	sourceCards.push_back(card);
}

vector<L1GctEmCand> L1GctElectronLeafCard::getOutput() {
	return finalSort.getOutput();
}