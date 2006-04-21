#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

L1GctJetLeafCard::L1GctJetLeafCard()
{
}

L1GctJetLeafCard::~L1GctJetLeafCard()
{
}

void L1GctJetLeafCard::addSource(L1GctSourceCard* card) {
	sourceCards.push_back(card);
}

void L1GctJetLeafCard::reset() {

}

void L1GctJetLeafCard::fetchInput() {

}

void L1GctJetLeafCard::process() {

}
