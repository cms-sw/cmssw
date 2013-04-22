#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetSorter.h"

//DEFINE STATICS
const int L1GctWheelJetFpga::MAX_JETS_OUT = 4;
const unsigned int L1GctWheelJetFpga::MAX_LEAF_CARDS = 3;
const unsigned int L1GctWheelJetFpga::MAX_JETS_PER_LEAF = L1GctJetLeafCard::MAX_JET_FINDERS * L1GctJetFinderBase::MAX_JETS_OUT;
const int L1GctWheelJetFpga::MAX_JETS_IN = L1GctWheelJetFpga::MAX_LEAF_CARDS * L1GctWheelJetFpga::MAX_JETS_PER_LEAF;

L1GctWheelJetFpga::L1GctWheelJetFpga(int id,
				     const std::vector<L1GctJetLeafCard*>& inputLeafCards) :
  L1GctProcessor(),
  m_id(id),
  m_inputLeafCards(inputLeafCards),
  m_centralJetSorter(new L1GctJetSorter()),
  m_forwardJetSorter(new L1GctJetSorter()),
  m_tauJetSorter(new L1GctJetSorter()),
  m_inputJets(MAX_JETS_IN),
  m_rawCentralJets(MAX_JETS_IN),
  m_rawForwardJets(MAX_JETS_IN),
  m_rawTauJets(MAX_JETS_IN),
  m_inputHx(MAX_LEAF_CARDS),
  m_inputHy(MAX_LEAF_CARDS),
  m_inputHfSums(MAX_LEAF_CARDS),
  m_centralJets(MAX_JETS_OUT),
  m_forwardJets(MAX_JETS_OUT),
  m_tauJets(MAX_JETS_OUT),
  m_outputHx(0), m_outputHy(0), m_outputHfSums(),
  m_outputHxPipe(), m_outputHyPipe()
{
  if (checkSetup()) {

    setupJetsVectors(0);  //Initialises all the jet vectors with jets of the correct type.

  } else {
    if (m_verbose) {
      edm::LogError("L1GctSetupError") << "L1GctWheelJetFpga has been incorrectly constructed";
    }
  }
}

bool L1GctWheelJetFpga::checkSetup() const
{
  bool result=true;
  
  //Check object construction is ok
  if(m_id < 0 || m_id > 1)
    {
      result = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctWheelJetFpga::L1GctWheelJetFpga() : Wheel Jet FPGA ID " << m_id << " has been incorrectly constructed!\n"
	  << "ID number should be between the range of 0 to 1\n";
      }
    } 
  
  if(m_inputLeafCards.size() != MAX_LEAF_CARDS)
    {
      result = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctWheelJetFpga::L1GctWheelJetFpga() : Wheel Jet FPGA ID " << m_id << " has been incorrectly constructed!\n"
	  << "This class needs " << MAX_LEAF_CARDS << " jet leaf card pointers, yet only " << m_inputLeafCards.size()
	  << " leaf card pointers are present.\n";
      }
    }
  
  for(unsigned int i = 0; i < MAX_LEAF_CARDS; ++i)
    {
      if(m_inputLeafCards.at(i) == 0)
	{
	  result = false;
	  if (m_verbose) {
	    edm::LogWarning("L1GctSetupError")
	      << "L1GctWheelJetFpga::L1GctWheelJetFpga() : Wheel Jet FPGA ID " << m_id << " has been incorrectly constructed!\n"
	      << "Leaf card pointer " << i << " has not been set!\n";
	  }
	}
    }
  return result;
}

L1GctWheelJetFpga::~L1GctWheelJetFpga()
{
  if (m_centralJetSorter != 0) delete m_centralJetSorter;
  if (m_forwardJetSorter != 0) delete m_forwardJetSorter;
  if (m_tauJetSorter != 0)     delete m_tauJetSorter;
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
  os << endl;
  return os;
}	

void L1GctWheelJetFpga::resetProcessor()
{
  for (unsigned int i=0; i<MAX_LEAF_CARDS; ++i)
  {
    m_inputHx.at(i).reset();
    m_inputHy.at(i).reset();
    m_inputHfSums.at(i).reset();
  }
  m_outputHx.reset();
  m_outputHy.reset();
  m_outputHfSums.reset();
}

void L1GctWheelJetFpga::setupObjects()
{
  setupJetsVectors(static_cast<int16_t>(bxAbs()));
}

void L1GctWheelJetFpga::resetPipelines()
{
  m_outputHxPipe.reset(numOfBx());
  m_outputHyPipe.reset(numOfBx());
}

void L1GctWheelJetFpga::fetchInput()
{
  if (checkSetup()) {
    //Get Jets
    for(unsigned short iLeaf = 0; iLeaf < MAX_LEAF_CARDS; ++iLeaf)
      {
	if (m_inputLeafCards.at(iLeaf) != 0) {  //check that the pointers have been set up!

	  storeJets(m_inputLeafCards.at(iLeaf)->getOutputJetsA(), iLeaf, 0);
	  storeJets(m_inputLeafCards.at(iLeaf)->getOutputJetsB(), iLeaf, L1GctJetFinderBase::MAX_JETS_OUT);
	  storeJets(m_inputLeafCards.at(iLeaf)->getOutputJetsC(), iLeaf, 2*L1GctJetFinderBase::MAX_JETS_OUT);
        
	  // Deal with the Ht inputs
	  m_inputHx.at(iLeaf) = m_inputLeafCards.at(iLeaf)->getOutputHx();
	  m_inputHy.at(iLeaf) = m_inputLeafCards.at(iLeaf)->getOutputHy();

	  // Deal with the Hf tower sum inputs
	  m_inputHfSums.at(iLeaf) = m_inputLeafCards.at(iLeaf)->getOutputHfSums();
	}
      }
  }
}

void L1GctWheelJetFpga::process()
{
  if (checkSetup()) {
    classifyJets();

    m_centralJetSorter->setJets(m_rawCentralJets);
    m_forwardJetSorter->setJets(m_rawForwardJets);
    m_tauJetSorter->setJets(m_rawTauJets);

    m_rawCentralJets = m_centralJetSorter->getSortedJets();
    m_rawForwardJets = m_forwardJetSorter->getSortedJets();
    m_rawTauJets     = m_tauJetSorter->getSortedJets();

    for(unsigned short iJet = 0; iJet < MAX_JETS_OUT; ++iJet)
      {
	m_centralJets.at(iJet) = m_rawCentralJets.at(iJet);
	m_forwardJets.at(iJet) = m_rawForwardJets.at(iJet);
	m_tauJets.at(iJet) = m_rawTauJets.at(iJet);
      }

    //Ht processing
    m_outputHx = m_inputHx.at(0) + m_inputHx.at(1) + m_inputHx.at(2);
    m_outputHy = m_inputHy.at(0) + m_inputHy.at(1) + m_inputHy.at(2);

    //Hf tower sums processing
    m_outputHfSums = m_inputHfSums.at(0) + m_inputHfSums.at(1) + m_inputHfSums.at(2);

    m_outputHxPipe.store( m_outputHx, bxRel());
    m_outputHyPipe.store( m_outputHy, bxRel());
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
      if (m_verbose) {
	edm::LogError("L1GctInputError")
	  << "L1GctWheelJetFpga::setInputJet() : In WheelJetFpga ID  " << m_id << ", inputted jet candidate " 
	  << i << " is outside input index range of 0 to " << (MAX_JETS_IN-1) << "\n";
      }
    }
}

/// get the Et sums in internal component format
std::vector< L1GctInternHtMiss > L1GctWheelJetFpga::getInternalHtMiss() const
{

  std::vector< L1GctInternHtMiss > result;
  for (int bx=0; bx<numOfBx(); bx++) {
    result.push_back( L1GctInternHtMiss::emulatorMissHtx( m_outputHxPipe.contents.at(bx).value(),
							  m_outputHxPipe.contents.at(bx).overFlow(),
							  static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternHtMiss::emulatorMissHty( m_outputHyPipe.contents.at(bx).value(),
							  m_outputHyPipe.contents.at(bx).overFlow(),
							  static_cast<int16_t> (bx-bxMin()) ) );
  }
  return result;
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
  
  unsigned short pos=0;
  // In the case of two jets of equal rank, the sort will take the lower priority.
  // This corresponds to the lower position in the array. In order to mimic the hardware
  // behaviour, the order of jets from the input leaf cards is maintained here.
  for(currentJet = m_inputJets.begin(); currentJet != m_inputJets.end(); ++currentJet, ++pos)
    {
      if (!currentJet->empty()) {
	if(currentJet->isForward())  //forward jet
	  {
	    m_rawForwardJets.at(pos) = *currentJet;
	  }
	else
	  {
	    if(currentJet->isCentral())  //central non-tau jet.
	      {
		m_rawCentralJets.at(pos) = *currentJet;
	      }
	    else  //must be central tau-jet
	      {
		if(currentJet->isTau())
		  {
		    m_rawTauJets.at(pos) = *currentJet;
		  }
		else
		  { //shouldn't get here!
		    if (m_verbose) {
		      edm::LogWarning("L1GctProcessingError")
			<< "Unclassified jet found by WheelJetFpga id " << m_id
			<< ". Jet details follow." << std::endl << *currentJet << std::endl;
		    }
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
  m_rawCentralJets.assign(MAX_JETS_IN, tempCen);
  m_rawTauJets.assign    (MAX_JETS_IN, tempTau);
  m_rawForwardJets.assign(MAX_JETS_IN, tempFwd);

  m_centralJets.assign(MAX_JETS_OUT, tempCen);
  m_tauJets.assign    (MAX_JETS_OUT, tempTau);
  m_forwardJets.assign(MAX_JETS_OUT, tempFwd);
}
