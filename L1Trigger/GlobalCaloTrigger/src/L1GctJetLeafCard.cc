#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "FWCore/Utilities/interface/Exception.h"

L1GctJetLeafCard::L1GctJetLeafCard(int id, int iphi) :
  m_id(id),
  m_sourceCards(MAX_SOURCE_CARDS),
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

void L1GctJetLeafCard::setInputSourceCard(int i, L1GctSourceCard* sc) 
{
  if (i > 0 && i < MAX_SOURCE_CARDS)
  {
    m_sourceCards[i] = sc;
  }
  else
  {
    throw cms::Exception("RangeError")
    << "In L1GctJetLeafCard, Source Card " << i << " is outside input range of 0 to "
    << (MAX_SOURCE_CARDS-1) << "\n";
  }
}

