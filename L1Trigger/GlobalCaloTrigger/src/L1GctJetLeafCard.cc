#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//DEFINE STATICS
const int L1GctJetLeafCard::MAX_JET_FINDERS = 3;  
const unsigned int L1GctJetLeafCard::MAX_SOURCE_CARDS = 15;

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
  
  //iphi is redundant
  if(phiPosition != m_id%3)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
    << "Argument iphi is " << phiPosition << ", should be " << (m_id%3) << " for this ID value \n";
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

std::ostream& operator << (std::ostream& s, const L1GctJetLeafCard& card)
{
  s << "===L1GctJetLeafCard===" << endl;
  s << "ID = " << card.m_id << endl;
  s << "i_phi = " << card.phiPosition << endl;;
  s << "No of Source Cards = " << card.m_sourceCards.size() << endl;
  for (unsigned i=0; i<card.m_sourceCards.size(); i++) {
    s << "SourceCard* " << i << " = " << card.m_sourceCards[i]<< endl;
  }
  s << "Ex " << card.m_exSum << endl;
  s << "Ey " << card.m_eySum << endl;
  s << "Et " << card.m_etSum << endl;
  s << "Ht " << card.m_htSum << endl;
  s << "JetFinder A : " << endl << (*card.m_jetFinderA);
  s << "JetFinder B : " << endl << (*card.m_jetFinderB); 
  s << "JetFinder C : " << endl << (*card.m_jetFinderC);
  s << endl;

  return s;
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
  vector<L1GctScalarEtVal> etStripSum(6);
  etStripSum[0] = m_jetFinderA->getEtStrip0();
  etStripSum[1] = m_jetFinderA->getEtStrip1();
  etStripSum[2] = m_jetFinderB->getEtStrip0();
  etStripSum[3] = m_jetFinderB->getEtStrip1();
  etStripSum[4] = m_jetFinderC->getEtStrip0();
  etStripSum[5] = m_jetFinderC->getEtStrip1();

  m_etSum.reset();
  m_exSum.reset();
  m_eySum.reset();

  for (unsigned i=0; i<6; ++i) {
    m_etSum = m_etSum + etStripSum[i];
    m_exSum = m_exSum + exComponent(etStripSum[i], (phiPosition*6+i));
    m_eySum = m_eySum + eyComponent(etStripSum[i], (phiPosition*6+i));
  }

  m_htSum =
    m_jetFinderA->getHt() +
    m_jetFinderB->getHt() +
    m_jetFinderC->getHt();
}

// PRIVATE MEMBER FUNCTIONS
// Given a strip Et sum, perform rotations by sine and cosine
// factors to find the corresponding Ex and Ey

L1GctEtComponent
L1GctJetLeafCard::exComponent(const L1GctScalarEtVal etStrip, const unsigned jphi) const {
  unsigned fact = (2*jphi+10) % 36;
  return rotateEtValue(etStrip, fact);
}

L1GctEtComponent
L1GctJetLeafCard::eyComponent(const L1GctScalarEtVal etStrip, const unsigned jphi) const {
  unsigned fact = (2*jphi+19) % 36;
  return rotateEtValue(etStrip, fact);
}

// Here is where the rotations are actually done
// Procedure suitable for implementation in hardware, using
// integer multiplication and bit shifting operations
L1GctEtComponent
L1GctJetLeafCard::rotateEtValue(const L1GctScalarEtVal etStrip, const unsigned fact) const {
  // These factors correspond to the sine of angles from -90 degrees to
  // 90 degrees in 10 degree steps, multiplied by 256 and written in 20 bits
  const int factors[19] = {0xfff00, 0xfff04, 0xfff10, 0xfff23, 0xfff3c,
			   0xfff5c, 0xfff80, 0xfffa9, 0xfffd4, 0x00000,
			   0x0002c, 0x00057, 0x00080, 0x000a4, 0x000c4,
			   0x000dd, 0x000f0, 0x000fc, 0x00100};
  const int maxEt=1<<(etStrip.size());
  int myValue, myFact;

  if (fact >= 36) {
    throw cms::Exception("L1GctProcessingError")
      << "L1GctJetLeafCard::rotateEtValue() has been called with factor number "
      << fact << "; should be less than 36 \n";
  } 

  // Choose the required multiplication factor
  if (fact>18) { myFact = factors[(36-fact)]; }
  else { myFact = factors[fact]; }

  // Multiply the 12-bit Et value by the 20-bit factor.
  // Discard the eight LSB and interpret the result as
  // a 12-bit twos complement integer.
  myValue = (static_cast<int>(etStrip.value())*myFact) >> 8;
  myValue = myValue & (maxEt-1);
  if (myValue >= (maxEt/2)) {
    myValue = myValue - maxEt;
  }

  L1GctEtComponent temp(myValue);
  temp.setOverFlow(temp.overFlow() || etStrip.overFlow());

  return temp;
}
