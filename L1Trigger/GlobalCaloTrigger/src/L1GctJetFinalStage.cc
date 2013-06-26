#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetSorter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::ostream;
using std::endl;

//DEFINE STATICS
const unsigned int L1GctJetFinalStage::MAX_WHEEL_FPGAS = 2;
const int L1GctJetFinalStage::MAX_JETS_IN = L1GctJetFinalStage::MAX_WHEEL_FPGAS*L1GctWheelJetFpga::MAX_JETS_OUT;
const int L1GctJetFinalStage::MAX_JETS_OUT = 4;


L1GctJetFinalStage::L1GctJetFinalStage(const std::vector<L1GctWheelJetFpga*>& wheelFpgas):
  L1GctProcessor(),
  m_wheelFpgas(wheelFpgas),
  m_centralJetSorter(new L1GctJetSorter()),
  m_forwardJetSorter(new L1GctJetSorter()),
  m_tauJetSorter(new L1GctJetSorter()),
  m_inputCentralJets(MAX_JETS_IN),
  m_inputForwardJets(MAX_JETS_IN),
  m_inputTauJets(MAX_JETS_IN),
  m_centralJets(MAX_JETS_OUT),
  m_forwardJets(MAX_JETS_OUT),
  m_tauJets(MAX_JETS_OUT),
  m_setupOk(true)
{
  if(m_wheelFpgas.size() != MAX_WHEEL_FPGAS)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctJetFinalStage::L1GctJetFinalStage() : Jet Final Stage instance has been incorrectly constructed!\n"
	  << "This class needs " << MAX_WHEEL_FPGAS << " wheel jet FPGA pointers, yet only " << m_wheelFpgas.size()
	  << " wheel jet FPGA pointers are present.\n";
      }
    }
  
  for(unsigned int i=0; i < MAX_WHEEL_FPGAS; ++i)
    {
      if(m_wheelFpgas.at(i) == 0)
	{
	  m_setupOk = false;
	  if (m_verbose) {
	    edm::LogWarning("L1GctSetupError")
	      << "L1GctJetFinalStage::L1GctJetFinalStage() : Jet Final Stage instance has been incorrectly constructed!\n"
	      << "Wheel jet FPGA pointer " << i << " has not been set!\n";
	  }
	}
    }  
  if (!m_setupOk && m_verbose) {
    edm::LogError("L1GctSetupError") << "L1GctJetFinalStage has been incorrectly constructed";
  }
}

L1GctJetFinalStage::~L1GctJetFinalStage()
{
  if (m_centralJetSorter != 0) delete m_centralJetSorter;
  if (m_forwardJetSorter != 0) delete m_forwardJetSorter;
  if (m_tauJetSorter != 0)     delete m_tauJetSorter;
}

std::ostream& operator << (std::ostream& os, const L1GctJetFinalStage& fpga)
{
  os << "===L1GctJetFinalStage===" << endl;
  os << "No. of Wheel Jet FPGAs " << fpga.m_wheelFpgas.size() << std::endl;
  for(unsigned i=0; i < fpga.m_wheelFpgas.size(); i++)
    {
      os << "WheelJetFpga* " << i << " = " << fpga.m_wheelFpgas.at(i) << endl;
    } 
  os << "No. of input central Jets " << fpga.m_inputCentralJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_inputCentralJets.size(); i++)
    {
      os << fpga.m_inputCentralJets.at(i);
    } 
  os << "No. of input forward Jets " << fpga.m_inputForwardJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_inputForwardJets.size(); i++)
    {
      os << fpga.m_inputForwardJets.at(i);
    } 
  os << "No. of raw tau Jets " << fpga.m_inputTauJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_inputTauJets.size(); i++)
    {
      os << fpga.m_inputTauJets.at(i);
    } 
  os << "No. of output central Jets " << fpga.m_centralJets.contents.size() << std::endl;
  for(unsigned i=0; i < fpga.m_centralJets.contents.size(); i++)
    {
      os << fpga.m_centralJets.contents.at(i);
    } 
  os << "No. of output forward Jets " << fpga.m_forwardJets.contents.size() << std::endl;
  for(unsigned i=0; i < fpga.m_forwardJets.contents.size(); i++)
    {
      os << fpga.m_forwardJets.contents.at(i);
    } 
  os << "No. of output tau Jets " << fpga.m_tauJets.contents.size() << std::endl;
  for(unsigned i=0; i < fpga.m_tauJets.contents.size(); i++)
    {
      os << fpga.m_tauJets.contents.at(i);
    } 
  os << endl;
  return os;
}

void L1GctJetFinalStage::resetProcessor() {
  //Clear all jet data
  m_inputCentralJets.clear();
  m_inputForwardJets.clear();
  m_inputTauJets.clear();
  //Resize the vectors
  m_inputCentralJets.resize(MAX_JETS_IN);
  m_inputForwardJets.resize(MAX_JETS_IN);
  m_inputTauJets.resize(MAX_JETS_IN);
}

void L1GctJetFinalStage::resetPipelines() {
  m_centralJets.reset(numOfBx());
  m_forwardJets.reset(numOfBx());
  m_tauJets.reset    (numOfBx());
}

void L1GctJetFinalStage::fetchInput()
{
  if (m_setupOk) {
    // We fetch and store the negative eta jets first. This ensures they have
    // higher priority when sorting equal rank jets.
    for(unsigned short iWheel=0; iWheel < MAX_WHEEL_FPGAS; ++iWheel)
      {
	storeJets(m_inputCentralJets, m_wheelFpgas.at(iWheel)->getCentralJets(), iWheel);
	storeJets(m_inputForwardJets, m_wheelFpgas.at(iWheel)->getForwardJets(), iWheel);
	storeJets(m_inputTauJets, m_wheelFpgas.at(iWheel)->getTauJets(), iWheel);
      }
  }
}

void L1GctJetFinalStage::process()
{
  if (m_setupOk) {
    //Process jets
    m_centralJetSorter->setJets(m_inputCentralJets);
    m_forwardJetSorter->setJets(m_inputForwardJets);
    m_tauJetSorter->setJets(m_inputTauJets);

    m_centralJets.store(m_centralJetSorter->getSortedJets(), bxRel());
    m_forwardJets.store(m_forwardJetSorter->getSortedJets(), bxRel());
    m_tauJets.store    (m_tauJetSorter->getSortedJets(),     bxRel());
  }
}

void L1GctJetFinalStage::setInputCentralJet(int i, L1GctJetCand jet)
{
  if( ((jet.isCentral() && jet.bx() == bxAbs()) || jet.empty())
      && (i >= 0 && i < MAX_JETS_IN))
  {
    m_inputCentralJets.at(i) = jet;
  }
}

void L1GctJetFinalStage::setInputForwardJet(int i, L1GctJetCand jet)
{
  if( ((jet.isForward() && jet.bx() == bxAbs()) || jet.empty())
     && (i >= 0 && i < MAX_JETS_IN))
  {
    m_inputForwardJets.at(i) = jet;
  }
}

void L1GctJetFinalStage::setInputTauJet(int i, L1GctJetCand jet)
{
  if( ((jet.isTau() && jet.bx() == bxAbs()) || jet.empty())
    && (i >= 0 && i < MAX_JETS_IN))
  {
    m_inputTauJets.at(i) = jet;
  }
}

void L1GctJetFinalStage::storeJets(JetVector& storageVector, JetVector jets, unsigned short iWheel)
{
  for(unsigned short iJet = 0; iJet < L1GctWheelJetFpga::MAX_JETS_OUT; ++iJet)
  {
    if (jets.at(iJet).bx() == bxAbs()) {
      storageVector.at((iWheel*L1GctWheelJetFpga::MAX_JETS_OUT) + iJet) = jets.at(iJet);
    }
  }
}
