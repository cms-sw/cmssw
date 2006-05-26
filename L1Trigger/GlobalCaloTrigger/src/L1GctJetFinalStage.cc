#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"

#include "FWCore/Utilities/interface/Exception.h"

using std::ostream;
using std::endl;

L1GctJetFinalStage::L1GctJetFinalStage(std::vector<L1GctWheelJetFpga*> wheelFpgas):
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
    if(m_wheelFpgas[i] == 0)
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
  os << "=== Jet Final Stage ===" << endl;
  os << "No. of Wheel Jet FPGAs " << fpga.m_wheelFpgas.size() << std::endl;
  for(unsigned i=0; i < fpga.m_wheelFpgas.size(); i++)
    {
      os << *fpga.m_wheelFpgas[i];
    } 
  os << "No. of input central Jets " << fpga.m_inputCentralJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_inputCentralJets.size(); i++)
    {
      os << fpga.m_inputCentralJets[i];
    } 
  os << "No. of input forward Jets " << fpga.m_inputForwardJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_inputForwardJets.size(); i++)
    {
      os << fpga.m_inputForwardJets[i];
    } 
  os << "No. of raw tau Jets " << fpga.m_inputTauJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_inputTauJets.size(); i++)
    {
      os << fpga.m_inputTauJets[i];
    } 
  os << "No. of output central Jets " << fpga.m_centralJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_centralJets.size(); i++)
    {
      os << fpga.m_centralJets[i];
    } 
  os << "No. of output forward Jets " << fpga.m_forwardJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_forwardJets.size(); i++)
    {
      os << fpga.m_forwardJets[i];
    } 
  os << "No. of output tau Jets " << fpga.m_tauJets.size() << std::endl;
  for(unsigned i=0; i < fpga.m_tauJets.size(); i++)
    {
      os << fpga.m_tauJets[i];
    } 
  return os;
}

void L1GctJetFinalStage::reset()
{
  //Clear all jet data
  m_inputCentralJets.clear();
  m_inputForwardJets.clear();
  m_inputTauJets.clear();
  m_centralJets.clear();
  m_forwardJets.clear();
  m_tauJets.clear();
  //Resize the vectors
  m_inputCentralJets.resize(MAX_JETS_IN);
  m_inputForwardJets.resize(MAX_JETS_IN);
  m_inputTauJets.resize(MAX_JETS_IN);
  m_centralJets.resize(MAX_JETS_OUT);
  m_forwardJets.resize(MAX_JETS_OUT);
  m_tauJets.resize(MAX_JETS_OUT);
}

void L1GctJetFinalStage::fetchInput()
{
  for(unsigned short iWheel=0; iWheel < MAX_WHEEL_FPGAS; ++iWheel)
  {
    storeJets(m_inputCentralJets, m_wheelFpgas[iWheel]->getCentralJets(), iWheel);
    storeJets(m_inputForwardJets, m_wheelFpgas[iWheel]->getForwardJets(), iWheel);
    storeJets(m_inputTauJets, m_wheelFpgas[iWheel]->getTauJets(), iWheel);
  }
}

void L1GctJetFinalStage::process()
{
  //Process jets
  sort(m_inputCentralJets.begin(), m_inputCentralJets.end(), L1GctJetCand::rankGreaterThan());
  sort(m_inputForwardJets.begin(), m_inputForwardJets.end(), L1GctJetCand::rankGreaterThan());
  sort(m_inputTauJets.begin(), m_inputTauJets.end(), L1GctJetCand::rankGreaterThan());

  for(unsigned short iJet = 0; iJet < MAX_JETS_OUT; ++iJet)
  {
    m_centralJets[iJet] = m_inputCentralJets[iJet];
    m_forwardJets[iJet] = m_inputForwardJets[iJet];
    m_tauJets[iJet] = m_inputTauJets[iJet];
  }  
}

void L1GctJetFinalStage::setInputCentralJet(int i, L1GctJetCand jet)
{
  if(i >= 0 && i < MAX_JETS_IN)
  {
    m_inputCentralJets[i] = jet;
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
  if(i >= 0 && i < MAX_JETS_IN)
  {
    m_inputForwardJets[i] = jet;
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
  if(i >= 0 && i < MAX_JETS_IN)
  {
    m_inputTauJets[i] = jet;
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
    storageVector[(iWheel*L1GctWheelJetFpga::MAX_JETS_OUT) + iJet] = jets[iJet];
  }
}
