#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "FWCore/Utilities/interface/Exception.h"

L1GctJetLeafCard::L1GctJetLeafCard(int id, int iphi) :
  m_id(id),
  m_sourceCards(MAX_SOURCE_CARDS),
  phiPosition(iphi)
{
  m_jetFinderA = new L1GctJetFinder(3*id);
  m_jetFinderB = new L1GctJetFinder(3*id+1);
  m_jetFinderC = new L1GctJetFinder(3*id+2);
}

L1GctJetLeafCard::~L1GctJetLeafCard()
{
  delete m_jetFinderA;
  delete m_jetFinderB;
  delete m_jetFinderC;
}

std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card)
{
  os << "JLC ID " << card.m_id << std::endl;
  os << "JetFinder A " << (*card.m_jetFinderA) << std::endl;
  os << "JetFinder B " << (*card.m_jetFinderB) << std::endl; 
  os << "JetFinder C " << (*card.m_jetFinderC) << std::endl;
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

void L1GctJetLeafCard::reset()
{
  m_jetFinderA->reset();
  m_jetFinderB->reset();
  m_jetFinderC->reset();
  m_exSum.reset();
  m_eySum.reset();
  m_etSum.reset();
  m_htSum.reset();
}

void L1GctJetLeafCard::fetchInput() {
  m_jetFinderA->fetchInput();
  m_jetFinderB->fetchInput();
  m_jetFinderC->fetchInput();
}

void L1GctJetLeafCard::process() {
  // Perform the jet finding
  m_jetFinderA->process();
  m_jetFinderB->process();
  m_jetFinderC->process();
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
      m_jetFinderA->setInputSourceCard(0, sc);
      m_jetFinderB->setInputSourceCard(2, sc);
      break;
    case 1 :
      m_jetFinderA->setInputSourceCard(1, sc);
      m_jetFinderB->setInputSourceCard(3, sc);
      break;
    case 2 :
      m_jetFinderB->setInputSourceCard(0, sc);
      m_jetFinderC->setInputSourceCard(2, sc);
      m_jetFinderA->setInputSourceCard(7, sc);
      break;
    case 3 :
      m_jetFinderB->setInputSourceCard(1, sc);
      m_jetFinderC->setInputSourceCard(3, sc);
      m_jetFinderA->setInputSourceCard(8, sc);
      break;
    case 4 :
      m_jetFinderC->setInputSourceCard(0, sc);
      m_jetFinderB->setInputSourceCard(7, sc);
      break;
    case 5 :
      m_jetFinderC->setInputSourceCard(1, sc);
      m_jetFinderB->setInputSourceCard(8, sc);
      break;
    case 6 :
      m_jetFinderC->setInputSourceCard(7, sc);
      break;
    case 7 :
      m_jetFinderC->setInputSourceCard(8, sc);
      break;
    case 8 :
      m_jetFinderA->setInputSourceCard(2, sc);
      break;
    case 9 :
      m_jetFinderA->setInputSourceCard(3, sc);
      break;
    case 10 :
      m_jetFinderA->setInputSourceCard(4, sc);
      break;
    case 11 :
      m_jetFinderA->setInputSourceCard(5, sc);
      m_jetFinderB->setInputSourceCard(4, sc);
      break;
    case 12 :
      m_jetFinderA->setInputSourceCard(6, sc);
      m_jetFinderB->setInputSourceCard(5, sc);
      m_jetFinderC->setInputSourceCard(4, sc);
      break;
    case 13 :
      m_jetFinderB->setInputSourceCard(6, sc);
      m_jetFinderC->setInputSourceCard(5, sc);
      break;
    case 14 :
      m_jetFinderC->setInputSourceCard(6, sc);
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

void L1GctJetLeafCard::setJetEtCalibrationLut(L1GctJetEtCalibrationLut* jetEtCalLut)
{
  m_jetFinderA->setJetEtCalibrationLut(jetEtCalLut);
  m_jetFinderB->setJetEtCalibrationLut(jetEtCalLut);
  m_jetFinderC->setJetEtCalibrationLut(jetEtCalLut);
}
