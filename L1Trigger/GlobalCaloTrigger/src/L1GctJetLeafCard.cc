#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
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
  m_hfSums.reset();
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
  vector< etTotalType > etStripSum(6);
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

  m_hfSums = 
    m_jetFinderA->getHfSums() +
    m_jetFinderB->getHfSums() +
    m_jetFinderC->getHfSums();

}

bool L1GctJetLeafCard::setupOk() const {
  return (m_jetFinderA->setupOk() &&
          m_jetFinderB->setupOk() &&
          m_jetFinderC->setupOk()); }

// get the jet output
L1GctJetFinderBase::JetVector
L1GctJetLeafCard::getOutputJetsA() const { return m_jetFinderA->getJets(); }  ///< Output jetfinder A jets (lowest jetFinder in phi)
L1GctJetFinderBase::JetVector
L1GctJetLeafCard::getOutputJetsB() const { return m_jetFinderB->getJets(); }  ///< Output jetfinder B jets (middle jetFinder in phi)
L1GctJetFinderBase::JetVector
L1GctJetLeafCard::getOutputJetsC() const { return m_jetFinderC->getJets(); }  ///< Ouptut jetfinder C jets (highest jetFinder in phi)

// PRIVATE MEMBER FUNCTIONS
// Given a strip Et sum, perform rotations by sine and cosine
// factors to find the corresponding Ex and Ey

L1GctJetLeafCard::etComponentType
L1GctJetLeafCard::exComponent(const L1GctJetLeafCard::etTotalType etStrip, const unsigned jphi) const {
  unsigned fact = (2*jphi+10) % 36;
  return rotateEtValue(etStrip, fact);
}

L1GctJetLeafCard::etComponentType
L1GctJetLeafCard::eyComponent(const L1GctJetLeafCard::etTotalType etStrip, const unsigned jphi) const {
  unsigned fact = (2*jphi+19) % 36;
  return rotateEtValue(etStrip, fact);
}

// Here is where the rotations are actually done
// Procedure suitable for implementation in hardware, using
// integer multiplication and bit shifting operations
L1GctJetLeafCard::etComponentType
L1GctJetLeafCard::rotateEtValue(const L1GctJetLeafCard::etTotalType etStrip, const unsigned fact) const {
  // These factors correspond to the sine of angles from -90 degrees to
  // 90 degrees in 10 degree steps, multiplied by 512 and written in 22 bits
  const int factors[19] = {0x3ffe00, 0x3ffe08, 0x3ffe1f, 0x3ffe45, 0x3ffe78,
                           0x3ffeb7, 0x3fff00, 0x3fff51, 0x3fffa7, 0x000000,
                           0x000059, 0x0000af, 0x000100, 0x000149, 0x000188,
                           0x0001bb, 0x0001e1, 0x0001f8, 0x000200};
  const int maxEt=1<<etComponentSize;
  int myValue, myFact;

  if (fact >= 36) {
    throw cms::Exception("L1GctProcessingError")
      << "L1GctJetLeafCard::rotateEtValue() has been called with factor number "
      << fact << "; should be less than 36 \n";
  } 

  // Choose the required multiplication factor
  if (fact>18) { myFact = factors[(36-fact)]; }
  else { myFact = factors[fact]; }

  // Multiply the 12-bit Et value by the 22-bit factor.
  // Discard the eight LSB and interpret the result as
  // a 14-bit twos complement integer.
  // Adjust the value to avoid truncation errors since these
  // accumulate and cause problems for the missing Et measurement.
  myValue = (( static_cast<int>(etStrip.value()) * myFact ) + 0x80)>>8;
  myValue = myValue & (maxEt-1);
  if (myValue >= (maxEt/2)) {
    myValue = myValue - maxEt;
  }

  etComponentType temp(myValue);
  temp.setOverFlow(temp.overFlow() || etStrip.overFlow());
  return temp;

}
