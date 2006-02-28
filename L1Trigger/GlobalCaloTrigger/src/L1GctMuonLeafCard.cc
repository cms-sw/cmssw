#include "../interface/L1GctMuonLeafCard.h"

L1GctMuonLeafCard::L1GctMuonLeafCard()
{
}

L1GctMuonLeafCard::~L1GctMuonLeafCard()
{
}

void L1GctMuonLeafCard::addSource(L1GctSourceCard* card) {
	sourceCards.push_back(card);
}
