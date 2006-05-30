#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

//DEFINE STATICS
const int L1GctWheelJetFpga::MAX_JETS_OUT = 4;
const unsigned int L1GctWheelJetFpga::MAX_LEAF_CARDS = 3;
const int L1GctWheelJetFpga::MAX_JETS_IN = L1GctWheelJetFpga::MAX_LEAF_CARDS * L1GctJetLeafCard::MAX_JET_FINDERS * L1GctJetFinder::MAX_JETS_OUT;
const int L1GctWheelJetFpga::MAX_RAW_CJETS = 36;
const int L1GctWheelJetFpga::MAX_RAW_FJETS = 18;
const int L1GctWheelJetFpga::MAX_RAW_TJETS = 36;


L1GctWheelJetFpga::L1GctWheelJetFpga(int id, vector<L1GctJetLeafCard*> inputLeafCards):
  m_id(id),
  m_inputLeafCards(inputLeafCards),
  m_inputJets(MAX_JETS_IN),
  m_rawCentralJets(MAX_RAW_CJETS),
  m_rawForwardJets(MAX_RAW_FJETS),
  m_inputHt(MAX_LEAF_CARDS),
  m_centralJets(MAX_JETS_OUT),
  m_forwardJets(MAX_JETS_OUT),
  m_tauJets(MAX_JETS_OUT),
  m_outputJc(12)
{
  setupRawTauJetsVec();  //sets up tau jet vector, but with tau-veto bits set to false.
  
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
    if(m_inputLeafCards[i] == 0)
    {
      throw cms::Exception("L1GctSetupError")
      << "L1GctWheelJetFpga::L1GctWheelJetFpga() : Wheel Jet FPGA ID " << m_id << " has been incorrectly constructed!\n"
      << "Leaf card pointer " << i << " has not been set!\n";
    }
  }
}

L1GctWheelJetFpga::~L1GctWheelJetFpga()
{
}

ostream& operator << (ostream& os, const L1GctWheelJetFpga& fpga)
{
  os << "===L1GctWheelJetFPGA===" << endl;
  os << "ID = " << fpga.m_id << endl;
  os << "No of Input Leaf Cards " << fpga.m_inputLeafCards.size() << endl;
  for(unsigned i=0; i < fpga.m_inputLeafCards.size(); i++)
    {
      os << "InputLeafCard* " << i << " = " << fpga.m_inputLeafCards[i] << endl;
    } 
//   os << "No. of Input Jets " << fpga.m_inputJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_inputJets.size(); i++)
//     {
//       os << fpga.m_inputJets[i];
//     } 
  os << "Input Ht " << endl;
  for(unsigned i=0; i < fpga.m_inputHt.size(); i++)
    {
      os << (fpga.m_inputHt[i]) << endl;
    } 
//   os << "No. of raw central Jets " << fpga.m_rawCentralJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_rawCentralJets.size(); i++)
//     {
//       os << fpga.m_rawCentralJets[i];
//     } 
//   os << "No. of raw forward Jets " << fpga.m_rawForwardJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_rawForwardJets.size(); i++)
//     {
//       os << fpga.m_rawForwardJets[i];
//     } 
//   os << "No. of raw tau Jets " << fpga.m_rawTauJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_rawTauJets.size(); i++)
//     {
//       os << fpga.m_rawTauJets[i];
//     } 
  os << "Output Ht " << fpga.m_outputHt << endl;
  os << "Output Jet count " << endl;
  for(unsigned i=0; i < fpga.m_outputJc.size(); i++)
    {
      os << "Jet count " << i << ": " << fpga.m_outputJc[i] << endl;
    } 
//   os << "No. of output central Jets " << fpga.m_centralJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_centralJets.size(); i++)
//     {
//       os << fpga.m_centralJets[i];
//     } 
//   os << "No. of output forward Jets " << fpga.m_forwardJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_forwardJets.size(); i++)
//     {
//       os << fpga.m_forwardJets[i];
//     } 
//   os << "No. of output tau Jets " << fpga.m_tauJets.size() << endl;
//   for(unsigned i=0; i < fpga.m_tauJets.size(); i++)
//     {
//       os << fpga.m_tauJets[i];
//     } 
  os << endl;
  return os;
}	

void L1GctWheelJetFpga::reset()
{
  m_inputJets.clear();
  m_inputJets.resize(MAX_JETS_IN);

  m_rawCentralJets.clear();
  m_rawCentralJets.resize(MAX_RAW_CJETS);
  m_rawForwardJets.clear();
  m_rawForwardJets.resize(MAX_RAW_FJETS);
  m_rawTauJets.clear();
  setupRawTauJetsVec();

  m_centralJets.clear();
  m_centralJets.resize(MAX_JETS_OUT);
  m_forwardJets.clear();
  m_forwardJets.resize(MAX_JETS_OUT);
  m_tauJets.clear();
  m_tauJets.resize(MAX_JETS_OUT);

  for (unsigned int i=0; i<MAX_LEAF_CARDS; ++i)
  {
    m_inputHt[i].reset();
  }
  m_outputHt.reset();
  for (int i=0; i<12; ++i)
  {
    m_outputJc[i].reset();
  }
}

void L1GctWheelJetFpga::fetchInput()
{
  //Get Jets
  for(unsigned short iLeaf = 0; iLeaf < MAX_LEAF_CARDS; ++iLeaf)
  {
    assert(m_inputLeafCards[iLeaf] != 0);  //check that the pointers have been set up!

    storeJets(m_inputLeafCards[iLeaf]->getOutputJetsA(), iLeaf, 0);
    storeJets(m_inputLeafCards[iLeaf]->getOutputJetsB(), iLeaf, MAX_JETS_IN);
    storeJets(m_inputLeafCards[iLeaf]->getOutputJetsC(), iLeaf, 2*MAX_JETS_IN);
        
    // Deal with the Ht inputs
    m_inputHt[iLeaf] = m_inputLeafCards[iLeaf]->getOutputHt();
  }
}

void L1GctWheelJetFpga::process()
{
  classifyJets();

  sort(m_rawCentralJets.begin(), m_rawCentralJets.end(), L1GctJetCand::rankGreaterThan());
  sort(m_rawForwardJets.begin(), m_rawForwardJets.end(), L1GctJetCand::rankGreaterThan());
  sort(m_rawTauJets.begin(), m_rawTauJets.end(), L1GctJetCand::rankGreaterThan());

  for(unsigned short iJet = 0; iJet < MAX_JETS_OUT; ++iJet)
  {
    m_centralJets[iJet] = m_rawCentralJets[iJet];
    m_forwardJets[iJet] = m_rawForwardJets[iJet];
    m_tauJets[iJet] = m_rawTauJets[iJet];
  }

  //Ht processing
  m_outputHt = m_inputHt[0] + m_inputHt[1] + m_inputHt[2];

  //Jet count processing (to be added)
    
}

void L1GctWheelJetFpga::setInputJet(int i, L1GctJetCand jet)
{
  if(i >=0 && i < MAX_JETS_IN)
  {
    m_inputJets[i] =  jet;
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
    m_inputHt[i].setValue(ht);
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
  for(unsigned short iJet = 0; iJet < L1GctJetFinder::MAX_JETS_OUT; ++iJet)
  {
    m_inputJets[iLeaf*MAX_JETS_IN/MAX_LEAF_CARDS + offset + iJet] = jets[iJet];
  }
}

void L1GctWheelJetFpga::classifyJets()
{
  //Clear the contents of all three of the  raw jet vectors
  m_rawCentralJets.clear();
  m_rawCentralJets.resize(MAX_RAW_CJETS);
  m_rawForwardJets.clear();
  m_rawForwardJets.resize(MAX_RAW_FJETS);
  m_rawTauJets.clear();
  setupRawTauJetsVec();

  //Holds which jet finder we are on, in phi (from 0 to 8).
  unsigned short int jetFinderIndex = 0;

  //counter to help in calculation of jetFinderIndex.
  unsigned short int inputJetIndex = 0;

  JetVector::iterator currentJet;  
  
  //counters for entering the classified jets into the different raw jet vectors
  unsigned short int numCJets = 0;
  unsigned short int numFJets = 0;
  unsigned short int numTJets = 0;

  for(currentJet = m_inputJets.begin(); currentJet != m_inputJets.end(); ++currentJet)
  {
    if(currentJet->eta() >= L1GctJetCand::LOCAL_ETA_HF_START)  //forward jet
    {
        m_rawForwardJets[numFJets++] = currentJet->convertToGlobalJet(jetFinderIndex, m_id);
    }
    else
    {
      if(currentJet->tauVeto() == true)  //central non-tau jet.
      {
        m_rawCentralJets[numCJets++] = currentJet->convertToGlobalJet(jetFinderIndex, m_id);
      }
      else  //must be central tau-jet
      {
        m_rawTauJets[numTJets++] = currentJet->convertToGlobalJet(jetFinderIndex, m_id);
      }
    }
    //move onto the next jet finder phi position every 6 jets
    if(++inputJetIndex % L1GctJetFinder::MAX_JETS_OUT == 0) { ++jetFinderIndex; }
  }
}

void L1GctWheelJetFpga::setupRawTauJetsVec()
{
  m_rawTauJets.resize(MAX_RAW_TJETS);

  //now need to set all tau veto bits to false.
  for(JetVector::iterator currentJet = m_rawTauJets.begin(); currentJet != m_rawTauJets.end(); ++currentJet)
  {
    currentJet->setTauVeto(false);
  }
}
