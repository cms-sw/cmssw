#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTdrJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHardwareJetFinder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

//DEFINE STATICS
const int L1GctJetLeafCard::MAX_JET_FINDERS = 3;  

L1GctJetLeafCard::L1GctJetLeafCard(int id, int iphi, jetFinderType jfType):
  L1GctProcessor(),
  m_id(id),
  m_whichJetFinder(jfType),
  phiPosition(iphi),
  m_exSum(0), m_eySum(0),
  m_hxSum(0), m_hySum(0),
  m_etSum(0), m_htSum(0),
  m_hfSums(),
  m_exSumPipe(), m_eySumPipe(),
  m_hxSumPipe(), m_hySumPipe(),
  m_etSumPipe(), m_htSumPipe(),
  m_hfSumsPipe(),
  m_ctorInputOk(true)
{
  //Check jetLeafCard setup
  if(m_id < 0 || m_id > 5)
    {
      m_ctorInputOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
	  << "ID number should be between the range of 0 to 5\n";
      } 
    }
  
  //iphi is redundant
  if(phiPosition != m_id%3)
    {
      m_ctorInputOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
	  << "Argument iphi is " << phiPosition << ", should be " << (m_id%3) << " for this ID value \n";
      } 
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

    m_ctorInputOk = false;
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	<< "L1GctJetLeafCard::L1GctJetLeafCard() : Jet Leaf Card ID " << m_id << " has been incorrectly constructed!\n"
	<< "Unrecognised jetFinder type " << m_whichJetFinder << ", cannot setup jetFinders\n";
    }

  }

  if (!m_ctorInputOk && m_verbose) {
    edm::LogError("L1GctSetupError") << "Jet Leaf Card ID " << m_id << " has been incorrectly constructed";
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
  std::vector<L1GctJetFinderBase*> jfNeighbours(2);

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
    m_ctorInputOk = false;
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	<< "L1GctJetLeafCard::setNeighbourLeafCards() : In Jet Leaf Card ID " << m_id 
	<< " size of input vector should be 2, but is in fact " << neighbours.size() << "\n";
    }
  }
}

std::ostream& operator << (std::ostream& s, const L1GctJetLeafCard& card)
{
  using std::endl;

  s << "===L1GctJetLeafCard===" << endl;
  s << "ID = " << card.m_id << endl;
  s << "i_phi = " << card.phiPosition << endl;;
  s << "Ex " << card.m_exSum << endl;
  s << "Ey " << card.m_eySum << endl;
  s << "Hx " << card.m_hxSum << endl;
  s << "Hy " << card.m_hySum << endl;
  s << "Et " << card.m_etSum << endl;
  s << "Ht " << card.m_htSum << endl;
  s << "JetFinder A : " << endl << (*card.m_jetFinderA);
  s << "JetFinder B : " << endl << (*card.m_jetFinderB); 
  s << "JetFinder C : " << endl << (*card.m_jetFinderC);
  s << endl;

  return s;
}

/// clear buffers
void L1GctJetLeafCard::reset() {
  L1GctProcessor::reset();
  m_jetFinderA->reset();
  m_jetFinderB->reset();
  m_jetFinderC->reset();
}

/// partially clear buffers
void L1GctJetLeafCard::setBxRange(const int firstBx, const int numberOfBx) {
  L1GctProcessor::setBxRange(firstBx, numberOfBx);
  m_jetFinderA->setBxRange(firstBx, numberOfBx);
  m_jetFinderB->setBxRange(firstBx, numberOfBx);
  m_jetFinderC->setBxRange(firstBx, numberOfBx);
}

void L1GctJetLeafCard::setNextBx(const int bx) {
  L1GctProcessor::setNextBx(bx);
  m_jetFinderA->setNextBx(bx);
  m_jetFinderB->setNextBx(bx);
  m_jetFinderC->setNextBx(bx);
}

void L1GctJetLeafCard::resetProcessor()
{
  m_exSum.reset();
  m_eySum.reset();
  m_hxSum.reset();
  m_hySum.reset();
  m_etSum.reset();
  m_htSum.reset();
  m_hfSums.reset();
}

void L1GctJetLeafCard::resetPipelines()
{
  m_exSumPipe.reset(numOfBx());
  m_eySumPipe.reset(numOfBx());
  m_hxSumPipe.reset(numOfBx());
  m_hySumPipe.reset(numOfBx());
  m_etSumPipe.reset(numOfBx());
  m_htSumPipe.reset(numOfBx());
  m_hfSumsPipe.reset(numOfBx());
}

void L1GctJetLeafCard::fetchInput() {
  m_jetFinderA->fetchInput();
  m_jetFinderB->fetchInput();
  m_jetFinderC->fetchInput();
}

void L1GctJetLeafCard::process() {

  // Check the setup
  if (setupOk()) {

    // Perform the jet finding
    m_jetFinderA->process();
    m_jetFinderB->process();
    m_jetFinderC->process();

    // Finish Et and Ht sums for the Leaf Card
    // First Et and missing Et
//     std::vector< etTotalType > etStripSum(6);
//     etStripSum.at(0) = m_jetFinderA->getEtStrip0();
//     etStripSum.at(1) = m_jetFinderA->getEtStrip1();
//     etStripSum.at(2) = m_jetFinderB->getEtStrip0();
//     etStripSum.at(3) = m_jetFinderB->getEtStrip1();
//     etStripSum.at(4) = m_jetFinderC->getEtStrip0();
//     etStripSum.at(5) = m_jetFinderC->getEtStrip1();

    m_etSum.reset();
    m_exSum.reset();
    m_eySum.reset();

//     for (unsigned i=0; i<6; ++i) {
//       m_etSum = m_etSum + etStripSum.at(i);
//     }

    m_etSum = m_etSum + m_jetFinderA->getEtSum();
    m_exSum = m_exSum + m_jetFinderA->getExSum();
    m_eySum = m_eySum + m_jetFinderA->getEySum();
    m_etSum = m_etSum + m_jetFinderB->getEtSum();
    m_exSum = m_exSum + m_jetFinderB->getExSum();
    m_eySum = m_eySum + m_jetFinderB->getEySum();
    m_etSum = m_etSum + m_jetFinderC->getEtSum();
    m_exSum = m_exSum + m_jetFinderC->getExSum();
    m_eySum = m_eySum + m_jetFinderC->getEySum();

//     for (unsigned i=0; i<3; ++i) {
//       unsigned jphi = 2*(phiPosition*3+i);
//       m_exSum = m_exSum + exComponent(etStripSum.at(2*i), etStripSum.at(2*i+1), jphi);
//       m_eySum = m_eySum + eyComponent(etStripSum.at(2*i), etStripSum.at(2*i+1), jphi);
//     }

//     std::cout << "ex sum " << m_exSum << ", from jetfinders " << exSumFromJf << std::endl;
//     std::cout << "ey sum " << m_eySum << ", from jetfinders " << eySumFromJf << std::endl;

    // Exactly the same procedure for Ht and missing Ht
    // Note using etTotalType for the strips but the output sum is etHadType
    std::vector< etTotalType > htStripSum(6);
    htStripSum.at(0) = m_jetFinderA->getHtStrip0();
    htStripSum.at(1) = m_jetFinderA->getHtStrip1();
    htStripSum.at(2) = m_jetFinderB->getHtStrip0();
    htStripSum.at(3) = m_jetFinderB->getHtStrip1();
    htStripSum.at(4) = m_jetFinderC->getHtStrip0();
    htStripSum.at(5) = m_jetFinderC->getHtStrip1();

    m_htSum.reset();
    m_hxSum.reset();
    m_hySum.reset();

    for (unsigned i=0; i<6; ++i) {
      m_htSum = m_htSum + htStripSum.at(i);
    }

    for (unsigned i=0; i<3; ++i) {
      unsigned jphi = 2*(phiPosition*3+i);
      m_hxSum = m_hxSum + hxComponent(htStripSum.at(2*i), htStripSum.at(2*i+1), jphi);
      m_hySum = m_hySum + hyComponent(htStripSum.at(2*i), htStripSum.at(2*i+1), jphi);
    }

    m_hfSums = 
      m_jetFinderA->getHfSums() +
      m_jetFinderB->getHfSums() +
      m_jetFinderC->getHfSums();

    // Store the outputs in pipelines
    m_exSumPipe.store  (m_exSum,  bxRel());
    m_eySumPipe.store  (m_eySum,  bxRel());
    m_hxSumPipe.store  (m_hxSum,  bxRel());
    m_hySumPipe.store  (m_hySum,  bxRel());
    m_etSumPipe.store  (m_etSum,  bxRel());
    m_htSumPipe.store  (m_htSum,  bxRel());
    m_hfSumsPipe.store (m_hfSums, bxRel());
  }
}

bool L1GctJetLeafCard::setupOk() const {
  return (m_ctorInputOk &&
	  m_jetFinderA->setupOk() &&
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
L1GctJetLeafCard::exComponent(const L1GctJetLeafCard::etTotalType etStrip0,
                              const L1GctJetLeafCard::etTotalType etStrip1,
			      const unsigned jphi) const {
  unsigned fact0 = (2*jphi+6) % 36;
  unsigned fact1 = (2*jphi+8) % 36;
  return etValueForJetFinder(etStrip0, fact0, etStrip1, fact1);
}

L1GctJetLeafCard::etComponentType
L1GctJetLeafCard::eyComponent(const L1GctJetLeafCard::etTotalType etStrip0,
                              const L1GctJetLeafCard::etTotalType etStrip1,
			      const unsigned jphi) const {
  unsigned fact0 = (2*jphi+15) % 36;
  unsigned fact1 = (2*jphi+17) % 36;
  return etValueForJetFinder(etStrip0, fact0, etStrip1, fact1);
}

// The same thing for Hx and Hy. Note that due to the exchange of data between jetfinders,
// the phi rotations are different for jet "strip sums" than for the raw region energies above

L1GctJetLeafCard::etComponentType
L1GctJetLeafCard::hxComponent(const L1GctJetLeafCard::etTotalType etStrip0,
                              const L1GctJetLeafCard::etTotalType etStrip1,
			      const unsigned jphi) const {
  unsigned fact0 = (2*jphi+10) % 36;
  unsigned fact1 = (2*jphi+ 4) % 36;
  return etValueForJetFinder(etStrip0, fact0, etStrip1, fact1);
}

L1GctJetLeafCard::etComponentType
L1GctJetLeafCard::hyComponent(const L1GctJetLeafCard::etTotalType etStrip0,
                              const L1GctJetLeafCard::etTotalType etStrip1,
			      const unsigned jphi) const {
  unsigned fact0 = (2*jphi+19) % 36;
  unsigned fact1 = (2*jphi+13) % 36;
  return etValueForJetFinder(etStrip0, fact0, etStrip1, fact1);
}

// Here is where the rotations are actually done
// Procedure suitable for implementation in hardware, using
// integer multiplication and bit shifting operations
L1GctJetLeafCard::etComponentType
L1GctJetLeafCard::etValueForJetFinder(const etTotalType etStrip0, const unsigned fact0,
                                      const etTotalType etStrip1, const unsigned fact1) const{
  // These factors correspond to the sine of angles from -90 degrees to
  // 90 degrees in 10 degree steps, multiplied by 16383 and written in 28 bits
  const int factors[19] = {0xfffc001, 0xfffc0fa, 0xfffc3dd, 0xfffc894, 0xfffcefa,
                           0xfffd6dd, 0xfffe000, 0xfffea1d, 0xffff4e3, 0x0000000,
                           0x0000b1d, 0x00015e3, 0x0002000, 0x0002923, 0x0003106,
                           0x000376c, 0x0003c23, 0x0003f06, 0x0003fff};

  static const int internalComponentSize=15;
  static const int maxEt=1<<internalComponentSize;

  int rotatedValue0, rotatedValue1, myFact;
  int etComponentSum = 0;

  if (fact0 >= 36 || fact1 >= 36) {
    if (m_verbose) {
      edm::LogError("L1GctProcessingError")
	<< "L1GctJetLeafCard::rotateEtValue() has been called with factor numbers "
	<< fact0 << " and " << fact1 << "; should be less than 36 \n";
    } 
  } else {

    // First strip - choose the required multiplication factor
    if (fact0>18) { myFact = factors[(36-fact0)]; }
    else { myFact = factors[fact0]; }

    // Multiply the 14-bit Et value by the 28-bit factor.
    rotatedValue0 = static_cast<int>(etStrip0.value()) * myFact;

    // Second strip - choose the required multiplication factor
    if (fact1>18) { myFact = factors[(36-fact1)]; }
    else { myFact = factors[fact1]; }

    // Multiply the 14-bit Et value by the 28-bit factor.
    rotatedValue1 = static_cast<int>(etStrip1.value()) * myFact;

    // Add the two scaled values together, with full resolution including
    // fractional parts from the sin(phi), cos(phi) scaling.
    // Adjust the value to avoid truncation errors since these
    // accumulate and cause problems for the missing Et measurement.
    // Then discard the 13 LSB and interpret the result as
    // a 15-bit twos complement integer.
    etComponentSum = ((rotatedValue0 + rotatedValue1) + 0x1000)>>13;

    etComponentSum = etComponentSum & (maxEt-1);
    if (etComponentSum >= (maxEt/2)) {
      etComponentSum = etComponentSum - maxEt;
    }
  }

  // Store as a TwosComplement format integer and return
  etComponentType temp(etComponentSum);
  temp.setOverFlow(temp.overFlow() || etStrip0.overFlow() || etStrip1.overFlow());
  return temp;
}
