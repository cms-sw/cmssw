#include "../interface/L1GctJetLeafCard.h"

L1GctJetLeafCard::L1GctJetLeafCard()
{
}

L1GctJetLeafCard::~L1GctJetLeafCard()
{
}

void L1GctJetLeafCard::addSource(L1GctSourceCard* card) {
	sourceCards.push_back(card);
}
