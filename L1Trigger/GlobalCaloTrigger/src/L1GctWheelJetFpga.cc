#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include "FWCore/Utilities/interface/Exception.h"

using std::vector;

L1GctWheelJetFpga::L1GctWheelJetFpga(int id):
  m_id(id),
  m_inputLeafCards(MAX_LEAF_CARDS),
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
}

L1GctWheelJetFpga::~L1GctWheelJetFpga()
{
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

  for (int i=0; i<MAX_LEAF_CARDS; ++i)
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
    m_inputHt[iLeaf] = m_inputLeafCards[iLeaf]->outputHt();
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

void L1GctWheelJetFpga::setInputLeafCard(int i, L1GctJetLeafCard* card)
{
  if(i >= 0 && i < MAX_LEAF_CARDS)
  {
    m_inputLeafCards[i] = card;
  }
  else
  {
    throw cms::Exception("RangeError")
    << "In L1GctWheelJetFpga, Jet Leaf Card " << i << " is outside input range of 0 to "
    << (MAX_LEAF_CARDS-1) << "\n";
  }
}

void L1GctWheelJetFpga::setInputJet(int i, L1GctJetCand jet)
{
  if(i >=0 && i < MAX_JETS_IN)
  {
    m_inputJets[i] =  jet;
  }
  else
  {
    throw cms::Exception("RangeError")
    << "In L1GctWheelJetFpga, Jet Candidate " << i << " is outside input range of 0 to "
    << (MAX_JETS_IN-1) << "\n";
  }
}

void L1GctWheelJetFpga::setInputHt (int i, unsigned ht)
{   
  assert(i >= 0 && i < MAX_LEAF_CARDS);
  m_inputHt[i].setValue(ht);
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
