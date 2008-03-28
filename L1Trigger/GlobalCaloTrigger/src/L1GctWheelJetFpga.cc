#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounter.h"
#include <cassert>

//DEFINE STATICS
const int L1GctWheelJetFpga::MAX_JETS_OUT = 4;
const unsigned int L1GctWheelJetFpga::MAX_LEAF_CARDS = 3;
const unsigned int L1GctWheelJetFpga::MAX_JETS_PER_LEAF = L1GctJetLeafCard::MAX_JET_FINDERS * L1GctJetFinderBase::MAX_JETS_OUT;
const int L1GctWheelJetFpga::MAX_JETS_IN = L1GctWheelJetFpga::MAX_LEAF_CARDS * L1GctWheelJetFpga::MAX_JETS_PER_LEAF;
const int L1GctWheelJetFpga::MAX_RAW_CJETS = 36;
const int L1GctWheelJetFpga::MAX_RAW_FJETS = 18;
const int L1GctWheelJetFpga::MAX_RAW_TJETS = 36;
const unsigned int L1GctWheelJetFpga::N_JET_COUNTERS = std::min(L1GctJetCounterSetup::MAX_JET_COUNTERS,
                                                                L1GctJetCounts::MAX_TRUE_COUNTS);


L1GctWheelJetFpga::L1GctWheelJetFpga(int id,
				     std::vector<L1GctJetLeafCard*> inputLeafCards) :
  L1GctProcessor(),
  m_id(id),
  m_inputLeafCards(inputLeafCards),
  m_jetCounters(N_JET_COUNTERS),
  m_inputJets(MAX_JETS_IN),
  m_rawCentralJets(MAX_RAW_CJETS),
  m_rawForwardJets(MAX_RAW_FJETS),
  m_inputHt(MAX_LEAF_CARDS),
  m_inputHfSums(MAX_LEAF_CARDS),
  m_centralJets(MAX_JETS_OUT),
  m_forwardJets(MAX_JETS_OUT),
  m_tauJets(MAX_JETS_OUT),
  m_outputHt(0), m_outputHfSums(),
  m_outputJc(N_JET_COUNTERS)
{
  checkSetup();

  // Initalise the jetCounters with null jetCounterLuts
  for (unsigned int i=0; i < N_JET_COUNTERS; i++) {
    m_jetCounters.at(i) = new L1GctJetCounter(((100*m_id)+i), m_inputLeafCards);
  }
}

void L1GctWheelJetFpga::checkSetup()
{
  setupJetsVectors(0);  //Initialises all the jet vectors with jets of the correct type.
  
  //Check object construction is ok
  if(m_id < 0 || m_id > 1)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctWheelJetFpga::L1GctWheelJetFpga() : Wheel Jet FPGA ID " << m_id << " has been incorrectly constructed!\n"
    << "ID number should be between the range of 0 to 1\n";
  } 
  
  if(m_inputLeafCards.size() != MAX_LEAF_CARDS)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctWheelJetFpga::L1GctWheelJetFpga() : Wheel Jet FPGA ID " << m_id << " has been incorrectly constructed!\n"
    << "This class needs " << MAX_LEAF_CARDS << " jet leaf card pointers, yet only " << m_inputLeafCards.size()
    << " leaf card pointers are present.\n";
  }
  
  for(unsigned int i = 0; i < MAX_LEAF_CARDS; ++i)
  {
    if(m_inputLeafCards.at(i) == 0)
    {
      throw cms::Exception("L1GctSetupError")
      << "L1GctWheelJetFpga::L1GctWheelJetFpga() : Wheel Jet FPGA ID " << m_id << " has been incorrectly constructed!\n"
      << "Leaf card pointer " << i << " has not been set!\n";
    }
  }
}

L1GctWheelJetFpga::~L1GctWheelJetFpga()
{
  for (unsigned int i=0; i < N_JET_COUNTERS; i++) {
    delete m_jetCounters.at(i);
  }
}

std::ostream& operator << (std::ostream& os, const L1GctWheelJetFpga& fpga)
{
  using std::endl;
  os << "===L1GctWheelJetFPGA===" << endl;
  os << "ID = " << fpga.m_id << endl;
  os << "No of Input Leaf Cards " << fpga.m_inputLeafCards.size() << endl;
  for(unsigned i=0; i < fpga.m_inputLeafCards.size(); i++)
    {
      os << "InputLeafCard* " << i << " = " << fpga.m_inputLeafCards.at(i) << endl;
    } 
//   os << "No. of Input Jets " << fpga.m_inputJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_inputJets.size(); i++)
//     {
//       os << fpga.m_inputJets.at(i);
//     } 
  os << "Input Ht " << endl;
  for(unsigned i=0; i < fpga.m_inputHt.size(); i++)
    {
      os << (fpga.m_inputHt.at(i)) << endl;
    } 
//   os << "No. of raw central Jets " << fpga.m_rawCentralJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_rawCentralJets.size(); i++)
//     {
//       os << fpga.m_rawCentralJets.at(i);
//     } 
//   os << "No. of raw forward Jets " << fpga.m_rawForwardJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_rawForwardJets.size(); i++)
//     {
//       os << fpga.m_rawForwardJets.at(i);
//     } 
//   os << "No. of raw tau Jets " << fpga.m_rawTauJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_rawTauJets.size(); i++)
//     {
//       os << fpga.m_rawTauJets.at(i);
//     } 
  os << "Output Ht " << fpga.m_outputHt << endl;
  os << "Output Jet count " << endl;
  for(unsigned i=0; i < fpga.m_outputJc.size(); i++)
    {
      os << "Jet count " << i << ": " << fpga.m_outputJc.at(i) << endl;
    } 
//   os << "No. of output central Jets " << fpga.m_centralJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_centralJets.size(); i++)
//     {
//       os << fpga.m_centralJets.at(i);
//     } 
//   os << "No. of output forward Jets " << fpga.m_forwardJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_forwardJets.size(); i++)
//     {
//       os << fpga.m_forwardJets.at(i);
//     } 
//   os << "No. of output tau Jets " << fpga.m_tauJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_tauJets.size(); i++)
//     {
//       os << fpga.m_tauJets.at(i);
//     }
  os << "Jet counters:" << endl; 
  for (unsigned int i=0; i < L1GctWheelJetFpga::N_JET_COUNTERS; i++) {
    os << *fpga.m_jetCounters.at(i) << endl;
  }
  os << endl;
  return os;
}	

/// clear buffers
void L1GctWheelJetFpga::reset() {
  L1GctProcessor::reset();
  for (unsigned int i=0; i<N_JET_COUNTERS; ++i)
  {
    m_jetCounters.at(i)->reset();
  }
}

/// partially clear buffers
void L1GctWheelJetFpga::setBxRange(const int firstBx, const int numberOfBx) {
  L1GctProcessor::setBxRange(firstBx, numberOfBx);
  for (unsigned int i=0; i<N_JET_COUNTERS; ++i)
  {
    m_jetCounters.at(i)->setBxRange(firstBx, numberOfBx);
  }
}

void L1GctWheelJetFpga::setNextBx(const int bx) {
  L1GctProcessor::setNextBx(bx);
  for (unsigned int i=0; i<N_JET_COUNTERS; ++i)
  {
    m_jetCounters.at(i)->setNextBx(bx);
  }
}

void L1GctWheelJetFpga::resetProcessor()
{
  for (unsigned int i=0; i<MAX_LEAF_CARDS; ++i)
  {
    m_inputHt.at(i).reset();
    m_inputHfSums.at(i).reset();
  }
  m_outputHt.reset();
  m_outputHfSums.reset();
  for (unsigned int i=0; i<N_JET_COUNTERS; ++i)
  {
    m_outputJc.at(i).reset();
  }
}

void L1GctWheelJetFpga::setupObjects()
{
  setupJetsVectors(static_cast<int16_t>(bxAbs()));
}

void L1GctWheelJetFpga::fetchInput()
{
  //Get Jets
  for(unsigned short iLeaf = 0; iLeaf < MAX_LEAF_CARDS; ++iLeaf)
  {
    assert(m_inputLeafCards.at(iLeaf) != 0);  //check that the pointers have been set up!

    storeJets(m_inputLeafCards.at(iLeaf)->getOutputJetsA(), iLeaf, 0);
    storeJets(m_inputLeafCards.at(iLeaf)->getOutputJetsB(), iLeaf, L1GctJetFinderBase::MAX_JETS_OUT);
    storeJets(m_inputLeafCards.at(iLeaf)->getOutputJetsC(), iLeaf, 2*L1GctJetFinderBase::MAX_JETS_OUT);
        
    // Deal with the Ht inputs
    m_inputHt.at(iLeaf) = m_inputLeafCards.at(iLeaf)->getOutputHt();

    // Deal with the Hf tower sum inputs
    m_inputHfSums.at(iLeaf) = m_inputLeafCards.at(iLeaf)->getOutputHfSums();
  }
  // Deal with the jet counters
  for (unsigned int i=0; i<N_JET_COUNTERS; i++) {
    // m_jetCounters.at(i)->fetchInput();
    //==============================================
    // For efficiency, provide our own list of jets to 
    // all the jet counters instead of allowing them
    // to fetch the jets from the jetfinder outputs

    m_jetCounters.at(i)->setJets(m_inputJets);

    //==============================================
  }
}

void L1GctWheelJetFpga::process()
{
  //setupJetsVectors();
  classifyJets();

  sort(m_rawCentralJets.begin(), m_rawCentralJets.end(), L1GctJetFinderBase::rankGreaterThan());
  sort(m_rawForwardJets.begin(), m_rawForwardJets.end(), L1GctJetFinderBase::rankGreaterThan());
  sort(m_rawTauJets.begin(), m_rawTauJets.end(), L1GctJetFinderBase::rankGreaterThan());

  for(unsigned short iJet = 0; iJet < MAX_JETS_OUT; ++iJet)
  {
    m_centralJets.at(iJet) = m_rawCentralJets.at(iJet);
    m_forwardJets.at(iJet) = m_rawForwardJets.at(iJet);
    m_tauJets.at(iJet) = m_rawTauJets.at(iJet);
  }

  //Ht processing
  m_outputHt = m_inputHt.at(0) + m_inputHt.at(1) + m_inputHt.at(2);

  //Hf tower sums processing
  m_outputHfSums = m_inputHfSums.at(0) + m_inputHfSums.at(1) + m_inputHfSums.at(2);

  //Jet count processing
  for (unsigned int i=0; i<N_JET_COUNTERS; i++) {
    m_jetCounters.at(i)->process();
    m_outputJc.at(i) = m_jetCounters.at(i)->getValue();
  }
    
}

void L1GctWheelJetFpga::setInputJet(int i, L1GctJetCand jet)
{
  if(i >=0 && i < MAX_JETS_IN)
  {
    m_inputJets.at(i) =  jet;
  }
  else
  {
    throw cms::Exception("L1GctInputError")
    << "L1GctWheelJetFpga::setInputJet() : In WheelJetFpga ID  " << m_id << ", inputted jet candidate " 
    << i << " is outside input index range of 0 to " << (MAX_JETS_IN-1) << "\n";
  }
}

void L1GctWheelJetFpga::setInputHt (int i, unsigned ht)
{   
  if(i >= 0 && i < static_cast<int>(MAX_LEAF_CARDS))
  {
    m_inputHt.at(i).setValue(ht);
  }
  else
  {
    throw cms::Exception("L1GctInputError")
    << "L1GctWheelJetFpga::setInputHt() : In WheelJetFpga ID  " << m_id << ", inputted Ht value " 
    << i << " is outside input index range of 0 to " << (MAX_LEAF_CARDS-1) << "\n";
  }
} 

void L1GctWheelJetFpga::storeJets(JetVector jets, unsigned short iLeaf, unsigned short offset)
{
  for(unsigned short iJet = 0; iJet < L1GctJetFinderBase::MAX_JETS_OUT; ++iJet)
  {
    m_inputJets.at(iLeaf*MAX_JETS_PER_LEAF + offset + iJet) = jets.at(iJet);
  }
}

void L1GctWheelJetFpga::classifyJets()
{
  JetVector::iterator currentJet;  
  
  //counters for entering the classified jets into the different raw jet vectors
  unsigned short int numCJets = 0;
  unsigned short int numFJets = 0;
  unsigned short int numTJets = 0;

  for(currentJet = m_inputJets.begin(); currentJet != m_inputJets.end(); ++currentJet)
  {
    if (!currentJet->empty()) {
      if(currentJet->isForward())  //forward jet
	{
         assert(numFJets<MAX_RAW_FJETS);
	  m_rawForwardJets.at(numFJets++) = *currentJet;
	}
      else
	{
	  if(currentJet->isCentral())  //central non-tau jet.
	    {
             assert(numCJets<MAX_RAW_CJETS);
	      m_rawCentralJets.at(numCJets++) = *currentJet;
	    }
	  else  //must be central tau-jet
	    {
	    if(currentJet->isTau())
	      {
             assert(numTJets<MAX_RAW_TJETS);
		m_rawTauJets.at(numTJets++) = *currentJet;
	      }
	    else
	      { //shouldn't get here!
		throw cms::Exception("L1GctProcessingError")
		  << "Unclassified jet found by WheelJetFpga id " << m_id
		  << ". Jet details follow." << std::endl << *currentJet << std::endl;
	      }
	    }
	}
    }
  }
}

void L1GctWheelJetFpga::setupJetsVectors(const int16_t bx)
{
  // Create empty jet candidates with the three different combinations
  // of flags, corresponding to central, forward and tau jets
  L1GctJetCand tempCen(0, 0, 0, false, false, (uint16_t) 0, (uint16_t) 0, bx);
  L1GctJetCand tempTau(0, 0, 0, true,  false, (uint16_t) 0, (uint16_t) 0, bx);
  L1GctJetCand tempFwd(0, 0, 0, false, true,  (uint16_t) 0, (uint16_t) 0, bx);

  // Initialize the jet vectors with copies of the appropriate empty jet type
  m_rawCentralJets.assign(MAX_RAW_CJETS, tempCen);
  m_rawTauJets.assign    (MAX_RAW_TJETS, tempTau);
  m_rawForwardJets.assign(MAX_RAW_FJETS, tempFwd);

  m_centralJets.assign(MAX_JETS_OUT, tempCen);
  m_tauJets.assign    (MAX_JETS_OUT, tempTau);
  m_forwardJets.assign(MAX_JETS_OUT, tempFwd);
}
