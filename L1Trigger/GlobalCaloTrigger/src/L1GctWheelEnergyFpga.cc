#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::vector;
using std::ostream;
using std::endl;

//DEFINE STATICS
const unsigned int L1GctWheelEnergyFpga::MAX_LEAF_CARDS = L1GctWheelJetFpga::MAX_LEAF_CARDS;

L1GctWheelEnergyFpga::L1GctWheelEnergyFpga(int id, const std::vector<L1GctJetLeafCard*>& leafCards) :
  L1GctProcessor(),
  m_id(id),
  m_inputLeafCards(leafCards),
  m_inputEx(MAX_LEAF_CARDS),
  m_inputEy(MAX_LEAF_CARDS),
  m_inputEt(MAX_LEAF_CARDS),
  m_inputHt(MAX_LEAF_CARDS),
  m_outputEx(0), m_outputEy(0), m_outputEt(0), m_outputHt(0),
  m_setupOk(true),
  m_outputExPipe(), m_outputEyPipe(), m_outputEtPipe(), m_outputHtPipe()
{
  //Check wheelEnergyFpga setup
  if(m_id != 0 && m_id != 1)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctWheelEnergyFpga::L1GctWheelEnergyFpga() : Wheel Energy Fpga ID " << m_id << " has been incorrectly constructed!\n"
	  << "ID number should be 0 or 1.\n";
      } 
    }
  
  if(m_inputLeafCards.size() != MAX_LEAF_CARDS)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctWheelEnergyFpga::L1GctWheelEnergyFpga() : Wheel Energy Fpga ID " << m_id << " has been incorrectly constructed!\n"
	  << "This class needs " << MAX_LEAF_CARDS << " leaf card pointers, yet only " << m_inputLeafCards.size()
	  << " leaf card pointers are present.\n";
      }
    }
  
  for(unsigned int i = 0; i < m_inputLeafCards.size(); ++i)
    {
      if(m_inputLeafCards.at(i) == 0)
	{
	  m_setupOk = false;
	  if (m_verbose) {
	    edm::LogWarning("L1GctSetupError")
	      << "L1GctWheelEnergyFpga::L1GctWheelEnergyFpga() : Wheel Energy Fpga ID " << m_id << " has been incorrectly constructed!\n"
	      << "Input Leaf card pointer " << i << " has not been set!\n";
	  }
	}
    }
  if (!m_setupOk && m_verbose) {
    edm::LogError("L1GctSetupError") << "L1GctWheelEnergyFpga has been incorrectly constructed";
  }
}

L1GctWheelEnergyFpga::~L1GctWheelEnergyFpga()
{
}

ostream& operator << (ostream& os, const L1GctWheelEnergyFpga& fpga)
{
  os << "===L1GctWheelEnergyFPGA===" << endl;
  os << "ID : " << fpga.m_id << endl;
  os << "No. of Input Leaf Cards " << fpga.m_inputLeafCards.size() << endl;
  for(unsigned i=0; i < fpga.m_inputLeafCards.size(); i++)
    {
      os << "LeafCard* " << i << " = " << fpga.m_inputLeafCards.at(i) << endl;
    } 
  os << "Input Ex " << endl;
  for(unsigned i=0; i < fpga.m_inputEx.size(); i++)
    {
      os << fpga.m_inputEx.at(i) << endl;
    } 
  os << "Input Ey " << endl;
  for(unsigned i=0; i < fpga.m_inputEy.size(); i++)
    {
      os << fpga.m_inputEy.at(i) << endl;
    } 
  os << "Input Et " << endl;
  for(unsigned i=0; i < fpga.m_inputEt.size(); i++)
    {
      os << fpga.m_inputEt.at(i) << endl;
    } 
  os << "Input Ht " << endl;
  for(unsigned i=0; i < fpga.m_inputHt.size(); i++)
    {
      os << (fpga.m_inputHt.at(i)) << endl;
    } 
  os << "Output Ex " << fpga.m_outputEx << endl;
  os << "Output Ey " << fpga.m_outputEy << endl;
  os << "Output Et " << fpga.m_outputEt << endl;
  os << "Output Ht " << fpga.m_outputHt << endl;
  os << endl;
  return os;
}

void L1GctWheelEnergyFpga::resetProcessor()
{
  for (unsigned int i=0; i<MAX_LEAF_CARDS; i++) {
    m_inputEx.at(i).reset();
    m_inputEy.at(i).reset();
    m_inputEt.at(i).reset();
    m_inputHt.at(i).reset();
  }
  m_outputEx.reset();
  m_outputEy.reset();
  m_outputEt.reset();
  m_outputHt.reset();
}

void L1GctWheelEnergyFpga::resetPipelines()
{
  m_outputExPipe.reset(numOfBx());
  m_outputEyPipe.reset(numOfBx());
  m_outputEtPipe.reset(numOfBx());
  m_outputHtPipe.reset(numOfBx());
}

void L1GctWheelEnergyFpga::fetchInput()
{
  if (m_setupOk) {
    // Fetch the output values from each of our input leaf cards.
    for (unsigned int i=0; i<MAX_LEAF_CARDS; i++) {
      m_inputEx.at(i) = m_inputLeafCards.at(i)->getOutputEx();
      m_inputEy.at(i) = m_inputLeafCards.at(i)->getOutputEy();
      m_inputEt.at(i) = m_inputLeafCards.at(i)->getOutputEt();
      m_inputHt.at(i) = m_inputLeafCards.at(i)->getOutputHt();
    }
  }
}

void L1GctWheelEnergyFpga::process()
{
  if (m_setupOk) {
    m_outputEx = m_inputEx.at(0) + m_inputEx.at(1) + m_inputEx.at(2);
    m_outputEy = m_inputEy.at(0) + m_inputEy.at(1) + m_inputEy.at(2);
    m_outputEt = m_inputEt.at(0) + m_inputEt.at(1) + m_inputEt.at(2);
    m_outputHt = m_inputHt.at(0) + m_inputHt.at(1) + m_inputHt.at(2);
    if (m_outputEt.overFlow()) m_outputEt.setValue(etTotalMaxValue);
    if (m_outputHt.overFlow()) m_outputHt.setValue(htTotalMaxValue);

    m_outputExPipe.store( m_outputEx, bxRel());
    m_outputEyPipe.store( m_outputEy, bxRel());
    m_outputEtPipe.store( m_outputEt, bxRel());
    m_outputHtPipe.store( m_outputHt, bxRel());
  }
}


///
/// set input data
void L1GctWheelEnergyFpga::setInputEnergy(unsigned i, int ex, int ey, unsigned et, unsigned ht)
{
  // Set the three input values from this Leaf card
  if (i<MAX_LEAF_CARDS) { // i >= 0, since i is unsigned
    m_inputEx.at(i).setValue(ex);
    m_inputEy.at(i).setValue(ey);
    m_inputEt.at(i).setValue(et);
    m_inputHt.at(i).setValue(ht);
  }

}

/// get the Et sums in internal component format
std::vector< L1GctInternEtSum  > L1GctWheelEnergyFpga::getInternalEtSums() const
{

  std::vector< L1GctInternEtSum > result;
  for (int bx=0; bx<numOfBx(); bx++) {
    result.push_back( L1GctInternEtSum::fromEmulatorMissEtxOrEty( m_outputExPipe.contents.at(bx).value(),
								  m_outputExPipe.contents.at(bx).overFlow(),
								  static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorMissEtxOrEty( m_outputEyPipe.contents.at(bx).value(),
								  m_outputEyPipe.contents.at(bx).overFlow(),
								  static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorTotalEtOrHt( m_outputEtPipe.contents.at(bx).value(),
								 m_outputEtPipe.contents.at(bx).overFlow(),
								 static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorTotalEtOrHt( m_outputHtPipe.contents.at(bx).value(),
								 m_outputHtPipe.contents.at(bx).overFlow(),
								 static_cast<int16_t> (bx-bxMin()) ) );
  }
  return result;
}
