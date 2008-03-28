#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

using std::ostream;
using std::endl;

//DEFINE STATICS
const unsigned int L1GctJetFinalStage::MAX_WHEEL_FPGAS = 2;
const int L1GctJetFinalStage::MAX_JETS_IN = L1GctJetFinalStage::MAX_WHEEL_FPGAS*L1GctWheelJetFpga::MAX_JETS_OUT;
const int L1GctJetFinalStage::MAX_JETS_OUT = 4;


L1GctJetFinalStage::L1GctJetFinalStage(std::vector<L1GctWheelJetFpga*> wheelFpgas):
  L1GctProcessor(),
  m_wheelFpgas(wheelFpgas),
  m_inputCentralJets(MAX_JETS_IN),
  m_inputForwardJets(MAX_JETS_IN),
  m_inputTauJets(MAX_JETS_IN),
  m_centralJets(MAX_JETS_OUT),
  m_forwardJets(MAX_JETS_OUT),
  m_tauJets(MAX_JETS_OUT)
{
  if(m_wheelFpgas.size() != MAX_WHEEL_FPGAS)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetFinalStage::L1GctJetFinalStage() : Jet Final Stage instance has been incorrectly constructed!\n"
    << "This class needs " << MAX_WHEEL_FPGAS << " wheel jet FPGA pointers, yet only " << m_wheelFpgas.size()
    << " wheel jet FPGA pointers are present.\n";
  }
  
  for(unsigned int i=0; i < MAX_WHEEL_FPGAS; ++i)
  {
    if(m_wheelFpgas.at(i) == 0)
    {
      throw cms::Exception("L1GctSetupError")
      << "L1GctJetFinalStage::L1GctJetFinalStage() : Jet Final Stage instance has been incorrectly constructed!\n"
      << "Wheel jet FPGA pointer " << i << " has not been set!\n";
    }
  }  
}

L1GctJetFinalStage::~L1GctJetFinalStage()
{
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
  for(unsigned short iWheel=0; iWheel < MAX_WHEEL_FPGAS; ++iWheel)
  {
    storeJets(m_inputCentralJets, m_wheelFpgas.at(iWheel)->getCentralJets(), iWheel);
    storeJets(m_inputForwardJets, m_wheelFpgas.at(iWheel)->getForwardJets(), iWheel);
    storeJets(m_inputTauJets, m_wheelFpgas.at(iWheel)->getTauJets(), iWheel);
  }
}

void L1GctJetFinalStage::process()
{
  //Process jets
  sort(m_inputCentralJets.begin(), m_inputCentralJets.end(), L1GctJetFinderBase::rankGreaterThan());
  sort(m_inputForwardJets.begin(), m_inputForwardJets.end(), L1GctJetFinderBase::rankGreaterThan());
  sort(m_inputTauJets.begin(), m_inputTauJets.end(), L1GctJetFinderBase::rankGreaterThan());

  //Copy data to output buffer
  m_centralJets.store(m_inputCentralJets, bxRel());
  m_forwardJets.store(m_inputForwardJets, bxRel());
  m_tauJets.store    (m_inputTauJets,     bxRel());
}

void L1GctJetFinalStage::setInputCentralJet(int i, L1GctJetCand jet)
{
  assert (jet.isCentral() && jet.bx() == bxAbs());
  if(i >= 0 && i < MAX_JETS_IN)
  {
    m_inputCentralJets.at(i) = jet;
  }
  else
  {
    throw cms::Exception("L1GctInputError")
    << "In L1GctJetFinalStage::setInputCentralJet() : Central Jet " << i
    << " is outside input range of 0 to " << (MAX_JETS_IN-1) << "\n";
  }
}

void L1GctJetFinalStage::setInputForwardJet(int i, L1GctJetCand jet)
{
  assert (jet.isForward() && jet.bx() == bxAbs());
  if(i >= 0 && i < MAX_JETS_IN)
  {
    m_inputForwardJets.at(i) = jet;
  }
  else
  {
    throw cms::Exception("L1GctInputError")
    << "In L1GctJetFinalStage::setInputForwardJet() : Forward Jet " << i
    << " is outside input range of 0 to " << (MAX_JETS_IN-1) << "\n";
  }
}

void L1GctJetFinalStage::setInputTauJet(int i, L1GctJetCand jet)
{
  assert (jet.isTau() && jet.bx() == bxAbs());
  if(i >= 0 && i < MAX_JETS_IN)
  {
    m_inputTauJets.at(i) = jet;
  }
  else
  {
    throw cms::Exception("L1GctInputError")
    << "In L1GctJetFinalStage::setInputTauJet() : Tau Jet " << i
    << " is outside input range of 0 to " << (MAX_JETS_IN-1) << "\n";
  }
}

void L1GctJetFinalStage::storeJets(JetVector& storageVector, JetVector jets, unsigned short iWheel)
{
  for(unsigned short iJet = 0; iJet < L1GctWheelJetFpga::MAX_JETS_OUT; ++iJet)
  {
    assert(jets.at(iJet).bx() == bxAbs());
    storageVector.at((iWheel*L1GctWheelJetFpga::MAX_JETS_OUT) + iJet) = jets.at(iJet);
  }
}
