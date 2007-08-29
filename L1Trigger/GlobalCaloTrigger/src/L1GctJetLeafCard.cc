#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTdrJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHardwareJetFinder.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//DEFINE STATICS
const int L1GctJetLeafCard::MAX_JET_FINDERS = 3;  

L1GctJetLeafCard::L1GctJetLeafCard(int id, int iphi, jetFinderType jfType):
  m_id(id),
  m_whichJetFinder(jfType),
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
  
  switch (m_whichJetFinder) {
  case tdrJetFinder :
    m_jetFinderA = new L1GctTdrJetFinder( 3*id );
    m_jetFinderB = new L1GctTdrJetFinder(3*id+1);
    m_jetFinderC = new L1GctTdrJetFinder(3*id+2);
    break;

  case hardwareJetFinder :
    m_jetFinderA = new L1GctHardwareJetFinder( 3*id );
    m_jetFinderB = new L1GctHardwareJetFinder(3*id+1);
    m_jetFinderC = new L1GctHardwareJetFinder(3*id+2);
    break;

  default :

    throw cms::Exception("L1GctSetupError")
      << "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
      << "Unrecognised jetFinder type " << m_whichJetFinder << ", cannot setup jetFinders\n";

  }

}

L1GctJetLeafCard::~L1GctJetLeafCard()
{
  delete m_jetFinderA;
  delete m_jetFinderB;
  delete m_jetFinderC;
}

/// set pointers to neighbours
void L1GctJetLeafCard::setNeighbourLeafCards(std::vector<L1GctJetLeafCard*> neighbours)
{
  vector<L1GctJetFinderBase*> jfNeighbours(2);

  if (neighbours.size()==2) {

    jfNeighbours.at(0) = neighbours.at(0)->getJetFinderC();
    jfNeighbours.at(1) = m_jetFinderB;
    m_jetFinderA->setNeighbourJetFinders(jfNeighbours);

    jfNeighbours.at(0) = m_jetFinderA;
    jfNeighbours.at(1) = m_jetFinderC;
    m_jetFinderB->setNeighbourJetFinders(jfNeighbours);

    jfNeighbours.at(0) = m_jetFinderB;
    jfNeighbours.at(1) = neighbours.at(1)->getJetFinderA();
    m_jetFinderC->setNeighbourJetFinders(jfNeighbours);

  } else {
    throw cms::Exception("L1GctSetupError")
      << "L1GctJetLeafCard::setNeighbourLeafCards() : In Jet Leaf Card ID " << m_id 
      << " size of input vector should be 2, but is in fact " << neighbours.size() << "\n";
  }
}

std::ostream& operator << (std::ostream& s, const L1GctJetLeafCard& card)
{
  s << "===L1GctJetLeafCard===" << endl;
  s << "ID = " << card.m_id << endl;
  s << "i_phi = " << card.phiPosition << endl;;
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

  // Check the setup
  assert(setupOk());

  // Perform the jet finding
  m_jetFinderA->process();
  m_jetFinderB->process();
  m_jetFinderC->process();

  // Finish Et and Ht sums for the Leaf Card
  vector< L1GctUnsignedInt<12> > etStripSum(6);
  etStripSum.at(0) = m_jetFinderA->getEtStrip0();
  etStripSum.at(1) = m_jetFinderA->getEtStrip1();
  etStripSum.at(2) = m_jetFinderB->getEtStrip0();
  etStripSum.at(3) = m_jetFinderB->getEtStrip1();
  etStripSum.at(4) = m_jetFinderC->getEtStrip0();
  etStripSum.at(5) = m_jetFinderC->getEtStrip1();

  m_etSum.reset();
  m_exSum.reset();
  m_eySum.reset();

  for (unsigned i=0; i<6; ++i) {
    m_etSum = m_etSum + etStripSum.at(i);
    m_exSum = m_exSum + exComponent(etStripSum.at(i), (phiPosition*6+i));
    m_eySum = m_eySum + eyComponent(etStripSum.at(i), (phiPosition*6+i));
  }

  m_htSum =
    m_jetFinderA->getHt() +
    m_jetFinderB->getHt() +
    m_jetFinderC->getHt();
}

// PRIVATE MEMBER FUNCTIONS
// Given a strip Et sum, perform rotations by sine and cosine
// factors to find the corresponding Ex and Ey

L1GctTwosComplement<12>
L1GctJetLeafCard::exComponent(const L1GctUnsignedInt<12> etStrip, const unsigned jphi) const {
  unsigned fact = (2*jphi+10) % 36;
  return rotateEtValue(etStrip, fact);
}

L1GctTwosComplement<12>
L1GctJetLeafCard::eyComponent(const L1GctUnsignedInt<12> etStrip, const unsigned jphi) const {
  unsigned fact = (2*jphi+19) % 36;
  return rotateEtValue(etStrip, fact);
}

// Here is where the rotations are actually done
// Procedure suitable for implementation in hardware, using
// integer multiplication and bit shifting operations
L1GctTwosComplement<12>
L1GctJetLeafCard::rotateEtValue(const L1GctUnsignedInt<12> etStrip, const unsigned fact) const {
  // These factors correspond to the sine of angles from -90 degrees to
  // 90 degrees in 10 degree steps, multiplied by 256 and written in 20 bits
  const int factors[19] = {0xfff00, 0xfff04, 0xfff0f, 0xfff22, 0xfff3c,
			   0xfff5b, 0xfff80, 0xfffa8, 0xfffd4, 0x00000,
			   0x0002c, 0x00058, 0x00080, 0x000a5, 0x000c4,
			   0x000de, 0x000f1, 0x000fc, 0x00100};
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
  // Adjust the value to avoid truncation errors since these
  // accumulate and cause problems for the missing Et measurement.
  myValue = (( static_cast<int>(etStrip.value()) * myFact ) + 0x80)>>8;
  myValue = myValue & (maxEt-1);
  if (myValue >= (maxEt/2)) {
    myValue = myValue - maxEt;
  }

  L1GctTwosComplement<12> temp(myValue);
  temp.setOverFlow(temp.overFlow() || etStrip.overFlow());

  return temp;

}
