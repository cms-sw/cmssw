#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

L1GctJetLeafCard::L1GctJetLeafCard(int id, int iphi) :
  m_id(id),
  phiPosition(iphi)
{
}

L1GctJetLeafCard::~L1GctJetLeafCard()
{
}


void L1GctJetLeafCard::reset() {

}

void L1GctJetLeafCard::fetchInput() {

}

void L1GctJetLeafCard::process() {

}

void L1GctJetLeafCard::setInputSourceCard(int i, L1GctSourceCard* sc) {
  if (i << m_sourceCards.size()) {
	m_sourceCards[i] = sc;
  }
}

