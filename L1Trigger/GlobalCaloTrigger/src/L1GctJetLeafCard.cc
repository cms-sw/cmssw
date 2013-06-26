#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTdrJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHardwareJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctNullJetFinder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

  case nullJetFinder :
    m_jetFinderA = new L1GctNullJetFinder( 3*id );
    m_jetFinderB = new L1GctNullJetFinder(3*id+1);
    m_jetFinderC = new L1GctNullJetFinder(3*id+2);
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
void L1GctJetLeafCard::setNeighbourLeafCards(const std::vector<L1GctJetLeafCard*>& neighbours)
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
    m_etSum =
      m_jetFinderA->getEtSum() +
      m_jetFinderB->getEtSum() +
      m_jetFinderC->getEtSum();
    if (m_etSum.overFlow()) m_etSum.setValue(etTotalMaxValue);
    m_exSum =
      ((etComponentType) m_jetFinderA->getExSum()) +
      ((etComponentType) m_jetFinderB->getExSum()) +
      ((etComponentType) m_jetFinderC->getExSum());
    m_eySum =
      ((etComponentType) m_jetFinderA->getEySum()) +
      ((etComponentType) m_jetFinderB->getEySum()) +
      ((etComponentType) m_jetFinderC->getEySum());

    // Exactly the same procedure for Ht and missing Ht
    m_htSum =
      m_jetFinderA->getHtSum() +
      m_jetFinderB->getHtSum() +
      m_jetFinderC->getHtSum();
    if (m_htSum.overFlow()) m_htSum.setValue(htTotalMaxValue);
    m_hxSum =
      ((htComponentType) m_jetFinderA->getHxSum()) +
      ((htComponentType) m_jetFinderB->getHxSum()) +
      ((htComponentType) m_jetFinderC->getHxSum());
    m_hySum =
      ((htComponentType) m_jetFinderA->getHySum()) +
      ((htComponentType) m_jetFinderB->getHySum()) +
      ((htComponentType) m_jetFinderC->getHySum());

    // And the same again for Hf Sums
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

/// get the Et sums in internal component format
std::vector< L1GctInternEtSum  > L1GctJetLeafCard::getInternalEtSums() const
{

  std::vector< L1GctInternEtSum > result;
  for (int bx=0; bx<numOfBx(); bx++) {
    result.push_back( L1GctInternEtSum::fromEmulatorJetTotEt ( m_etSumPipe.contents.at(bx).value(),
							       m_etSumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorJetMissEt( m_exSumPipe.contents.at(bx).value(),
							       m_exSumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorJetMissEt( m_eySumPipe.contents.at(bx).value(),
							       m_eySumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorJetTotHt ( m_htSumPipe.contents.at(bx).value(),
							       m_htSumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
  }
  return result;
}

std::vector< L1GctInternHtMiss > L1GctJetLeafCard::getInternalHtMiss() const
{

  std::vector< L1GctInternHtMiss > result;
  for (int bx=0; bx<numOfBx(); bx++) {
    result.push_back( L1GctInternHtMiss::emulatorMissHtxHty( m_hxSumPipe.contents.at(bx).value(),
							     m_hySumPipe.contents.at(bx).value(),
							     m_hxSumPipe.contents.at(bx).overFlow(),
							     static_cast<int16_t> (bx-bxMin()) ) );
  }
  return result;

}
