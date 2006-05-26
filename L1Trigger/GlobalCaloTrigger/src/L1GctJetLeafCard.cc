#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

L1GctJetLeafCard::L1GctJetLeafCard(int id, int iphi, vector<L1GctSourceCard*> sourceCards,
                                   L1GctJetEtCalibrationLut* jetEtCalLut):
  m_id(id),
  m_sourceCards(sourceCards),
  phiPosition(iphi)
{
  //Check jetLeafCard setup
  if(m_id < 0 || m_id > 5)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
    << "ID number should be between the range of 0 to 5\n";
  } 
  
  if(m_sourceCards.size() != MAX_SOURCE_CARDS)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
    << "This class needs " << MAX_SOURCE_CARDS << " source card pointers, yet only " << m_sourceCards.size()
    << " source card pointers are present.\n";
  }
  
  for(unsigned int i = 0; i < m_sourceCards.size(); ++i)
  {
    if(m_sourceCards[i] == 0)
    {
      throw cms::Exception("L1GctSetupError")
      << "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
      << "Source card pointer " << i << " has not been set!\n";
    }
  }

  //create vectors to pass into the three jetfinders
  vector<L1GctSourceCard*> srcCardsForJetFinderA(L1GctJetFinder::MAX_SOURCE_CARDS);
  vector<L1GctSourceCard*> srcCardsForJetFinderB(L1GctJetFinder::MAX_SOURCE_CARDS);
  vector<L1GctSourceCard*> srcCardsForJetFinderC(L1GctJetFinder::MAX_SOURCE_CARDS);

  srcCardsForJetFinderA[0] = m_sourceCards[0];
  srcCardsForJetFinderA[1] = m_sourceCards[1];
  srcCardsForJetFinderA[2] = m_sourceCards[8];
  srcCardsForJetFinderA[3] = m_sourceCards[9];
  srcCardsForJetFinderA[4] = m_sourceCards[10];
  srcCardsForJetFinderA[5] = m_sourceCards[11];
  srcCardsForJetFinderA[6] = m_sourceCards[12];
  srcCardsForJetFinderA[7] = m_sourceCards[2];
  srcCardsForJetFinderA[8] = m_sourceCards[3];
  
  srcCardsForJetFinderB[0] = m_sourceCards[2];
  srcCardsForJetFinderB[1] = m_sourceCards[3];
  srcCardsForJetFinderB[2] = m_sourceCards[0];
  srcCardsForJetFinderB[3] = m_sourceCards[1];
  srcCardsForJetFinderB[4] = m_sourceCards[11];
  srcCardsForJetFinderB[5] = m_sourceCards[12];
  srcCardsForJetFinderB[6] = m_sourceCards[13];
  srcCardsForJetFinderB[7] = m_sourceCards[4];
  srcCardsForJetFinderB[8] = m_sourceCards[5];

  srcCardsForJetFinderC[0] = m_sourceCards[4];
  srcCardsForJetFinderC[1] = m_sourceCards[5];
  srcCardsForJetFinderC[2] = m_sourceCards[2];
  srcCardsForJetFinderC[3] = m_sourceCards[3];
  srcCardsForJetFinderC[4] = m_sourceCards[12];
  srcCardsForJetFinderC[5] = m_sourceCards[13];
  srcCardsForJetFinderC[6] = m_sourceCards[14];
  srcCardsForJetFinderC[7] = m_sourceCards[6];
  srcCardsForJetFinderC[8] = m_sourceCards[7];
  
  m_jetFinderA = new L1GctJetFinder(3*id, srcCardsForJetFinderA, jetEtCalLut);
  m_jetFinderB = new L1GctJetFinder(3*id+1, srcCardsForJetFinderB, jetEtCalLut);
  m_jetFinderC = new L1GctJetFinder(3*id+2, srcCardsForJetFinderC, jetEtCalLut);
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
