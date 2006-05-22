#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "FWCore/Utilities/interface/Exception.h"

L1GctJetLeafCard::L1GctJetLeafCard(int id, int iphi) :
  m_id(id),
  m_sourceCards(MAX_SOURCE_CARDS),
  phiPosition(iphi)
{
  jetFinderA = new L1GctJetFinder(3*id);
  jetFinderB = new L1GctJetFinder(3*id+1);
  jetFinderC = new L1GctJetFinder(3*id+2);
}

L1GctJetLeafCard::~L1GctJetLeafCard()
{
}

std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card)
{
  os << "JLC ID " << card.m_id << std::endl;
  os << "JetFinder A " << (*card.jetFinderA) << std::endl;
  os << "JetFinder B " << (*card.jetFinderB) << std::endl; 
  os << "JetFinder C " << (*card.jetFinderC) << std::endl;
  os << "Phi " << card.phiPosition << std::endl;;
  os << "Ex " << card.m_exSum;
  os << "Ey " << card.m_eySum;
  os << "Et " << card.m_etSum;
  os << "Ht " << card.m_htSum;
  os << "No. of Source cards " << card.m_sourceCards.size() << std::endl;
  for(unsigned i=0; i < card.m_sourceCards.size(); i++)
    {
      if (card.m_sourceCards[i]!=0) os << (*card.m_sourceCards[i]); // These can be NULL!
    }
  return os;
}

void L1GctJetLeafCard::reset() {

}

void L1GctJetLeafCard::fetchInput() {
  jetFinderA->fetchInput();
  jetFinderB->fetchInput();
  jetFinderC->fetchInput();
}

void L1GctJetLeafCard::process() {
  // Perform the jet finding
  jetFinderA->process();
  jetFinderB->process();
  jetFinderC->process();
  // Finish Et and Ht sums for the Leaf Card
}

void L1GctJetLeafCard::setInputSourceCard(int i, L1GctSourceCard* sc) 
{
  if (i >= 0 && i < MAX_SOURCE_CARDS)
  {
    m_sourceCards[i] = sc;
    // setup the connections to the jetFinders
    // This is for the TDR algorithm currently
    // The hardware jetfinder will use only four
    // source cards, numbers 0,1,7 and 8.
    switch (i) {
    case 0 :
      jetFinderA->setInputSourceCard(0, sc);
      jetFinderB->setInputSourceCard(2, sc);
      break;
    case 1 :
      jetFinderA->setInputSourceCard(1, sc);
      jetFinderB->setInputSourceCard(3, sc);
      break;
    case 2 :
      jetFinderB->setInputSourceCard(0, sc);
      jetFinderC->setInputSourceCard(2, sc);
      jetFinderA->setInputSourceCard(7, sc);
      break;
    case 3 :
      jetFinderB->setInputSourceCard(1, sc);
      jetFinderC->setInputSourceCard(3, sc);
      jetFinderA->setInputSourceCard(8, sc);
      break;
    case 4 :
      jetFinderC->setInputSourceCard(0, sc);
      jetFinderB->setInputSourceCard(7, sc);
      break;
    case 5 :
      jetFinderC->setInputSourceCard(1, sc);
      jetFinderB->setInputSourceCard(8, sc);
      break;
    case 6 :
      jetFinderC->setInputSourceCard(7, sc);
      break;
    case 7 :
      jetFinderC->setInputSourceCard(8, sc);
      break;
    case 8 :
      jetFinderA->setInputSourceCard(2, sc);
      break;
    case 9 :
      jetFinderA->setInputSourceCard(3, sc);
      break;
    case 10 :
      jetFinderA->setInputSourceCard(4, sc);
      break;
    case 11 :
      jetFinderA->setInputSourceCard(5, sc);
      jetFinderB->setInputSourceCard(4, sc);
      break;
    case 12 :
      jetFinderA->setInputSourceCard(6, sc);
      jetFinderB->setInputSourceCard(5, sc);
      jetFinderC->setInputSourceCard(4, sc);
      break;
    case 13 :
      jetFinderB->setInputSourceCard(6, sc);
      jetFinderC->setInputSourceCard(5, sc);
      break;
    case 14 :
      jetFinderC->setInputSourceCard(6, sc);
      break;
    }
  }
  else
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetLeafCard::setInputSourceCard() : Source Card " << i << " is outside input range of 0 to "
    << (MAX_SOURCE_CARDS-1) << "\n";
  }
}

