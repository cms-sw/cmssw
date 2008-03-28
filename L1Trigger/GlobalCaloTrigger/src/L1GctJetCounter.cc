#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounter.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounterLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
 
#include "FWCore/Utilities/interface/Exception.h"  

//DEFINE STATICS
const unsigned int L1GctJetCounter::MAX_JETLEAF_CARDS = L1GctWheelJetFpga::MAX_LEAF_CARDS;
const unsigned int L1GctJetCounter::MAX_JETS_PER_LEAF = L1GctWheelJetFpga::MAX_JETS_PER_LEAF;
const unsigned int L1GctJetCounter::MAX_JETS_TO_COUNT = L1GctJetCounter::MAX_JETLEAF_CARDS*
                                                        L1GctJetCounter::MAX_JETS_PER_LEAF;


L1GctJetCounter::L1GctJetCounter(int id, std::vector<L1GctJetLeafCard*> leafCards,
                               L1GctJetCounterLut* jetCounterLut):
  L1GctProcessor(),
  m_id(id),
  m_jetLeafCards(leafCards),
  m_jets(MAX_JETS_TO_COUNT)
{
  //Check jetfinder setup
  if(m_id < 0 || m_id%100 >= 12 || m_id/100 >= 2)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetCounter::L1GctJetCounter() : Jet Counter ID " << m_id << " has been incorrectly constructed!\n"
    << "ID number should be between the range of 0 to 11, or 100 to 111\n";
  } 
  
  if(m_jetLeafCards.size() != MAX_JETLEAF_CARDS)
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetCounter::L1GctJetCounter() : Jet Counter ID " << m_id << " has been incorrectly constructed!\n"
    << "This class needs " << MAX_JETLEAF_CARDS << " leaf card pointers, yet only " << m_jetLeafCards.size()
    << " leaf card pointers are present.\n";
  }
  
  for(unsigned int i = 0; i < m_jetLeafCards.size(); ++i)
  {
    if(m_jetLeafCards.at(i) == 0)
    {
      throw cms::Exception("L1GctSetupError")
      << "L1GctJetCounter::L1GctJetCounter() : Jet Counter ID " << m_id << " has been incorrectly constructed!\n"
      << "Leaf card pointer " << i << " has not been set!\n";
    }
  }
  
  if(jetCounterLut == 0)
  {
    m_jetCounterLut = new L1GctJetCounterLut();
  } else {
    m_jetCounterLut = new L1GctJetCounterLut(*jetCounterLut);
  }
}

L1GctJetCounter::~L1GctJetCounter()
{
  delete m_jetCounterLut;
}

std::ostream& operator << (std::ostream& os, const L1GctJetCounter& algo)
{
  // Try to make this compact since it is probably
  // sitting inside a loop over 12 JetCounter instances
  os << "===L1GctJetCounter===" << std::endl;
  if ((algo.m_id/100) == 0) { os << "Minus wheel, "; }
  else { os << "Plus wheel, "; }
  os << "jet counter no. " << algo.m_id%100 <<  "; ID = " << algo.m_id << std::endl;
//   os << "JetCounterLut* = " <<  algo.m_jetCounterLut << std::endl;
  os << *algo.m_jetCounterLut << std::endl;
//   os << "No of Leaf cards " << algo.m_jetLeafCards.size();
//   os << ". Total input jets " << algo.m_jets.size() << std::endl;
//   for (unsigned i=0; i<algo.m_jetLeafCards.size(); i++) {
//     // One line of printing per leaf card
//     os << "JetLeafCard* " << i << " = " << algo.m_jetLeafCards.at(i);
//     os << " No of jets " << algo.m_jetLeafCards.at(i)->getOutputJetsA().size() ;
//     os << " + " << algo.m_jetLeafCards.at(i)->getOutputJetsB().size();
//     os << " + " << algo.m_jetLeafCards.at(i)->getOutputJetsC().size() << std::endl;
//   }
//   for(unsigned i=0; i < algo.m_jets.size(); ++i)
//     {
//       os << algo.m_jets.at(i); 
//     }
  os << "Current value of counter " << algo.m_value << std::endl;

  return os;
}


void L1GctJetCounter::resetProcessor()
{
  m_jets.clear();
  m_jets.resize(MAX_JETS_TO_COUNT);
  m_value.reset();
}

// Load the m_jets vector
void L1GctJetCounter::fetchInput()
{
  int jetnum=0;
  for (unsigned i=0; i<m_jetLeafCards.size(); i++) {
    if (jetnum+MAX_JETS_PER_LEAF>m_jets.size()) {
      throw cms::Exception("L1GctProcessingError")
	<< "L1GctJetCounter id= " << m_id << " trying to input too many jets for Leaf Card number " << i << std::endl
	<< "current jetnum is " << jetnum << " about to add " << MAX_JETS_PER_LEAF << std::endl;
    }
    L1GctJetLeafCard* jlc = m_jetLeafCards.at(i);
    for (unsigned j=0; j<L1GctJetFinderBase::MAX_JETS_OUT; j++) {
      m_jets.at(jetnum++) = jlc->getOutputJetsA().at(j);
      m_jets.at(jetnum++) = jlc->getOutputJetsB().at(j);
      m_jets.at(jetnum++) = jlc->getOutputJetsC().at(j);
    }
  }
}

/// set a new lut for this counter
void L1GctJetCounter::setLut(const L1GctJetCounterLut& lut)
{
  // Get rid of the old one
  delete m_jetCounterLut;
  m_jetCounterLut = new L1GctJetCounterLut(lut);
}

/// set a new lut for this counter by specifying the cuts - just one cut
void L1GctJetCounter::setLut(const L1GctJetCounterSetup::cutDescription& cut)
{
  // Get rid of the old one
  delete m_jetCounterLut;
  m_jetCounterLut = new L1GctJetCounterLut(cut);
}

/// set a new lut for this counter by specifying the cuts - list of cuts
void L1GctJetCounter::setLut(const L1GctJetCounterSetup::cutsListForJetCounter& cutList)
{
  // Get rid of the old one
  delete m_jetCounterLut;
  m_jetCounterLut = new L1GctJetCounterLut(cutList);
}

/// set the m_jets vector for test purposes
void L1GctJetCounter::setJets(JetVector& jets)
{
  m_jets = jets;
}

/// Count the jets passing cuts
void L1GctJetCounter::process() 
{
  for (unsigned i=0; i<m_jets.size(); i++) {
    if (m_jetCounterLut->passesCut(m_jets.at(i))) { m_value++; }
  }
}
