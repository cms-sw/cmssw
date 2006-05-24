#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

L1GctWheelEnergyFpga::L1GctWheelEnergyFpga(int id) :
	m_id(id),
        m_inputLeafCards(3),
	m_inputEx(3),
	m_inputEy(3),
	m_inputEt(3)
{
}

L1GctWheelEnergyFpga::~L1GctWheelEnergyFpga()
{
}

std::ostream& operator << (std::ostream& os, const L1GctWheelEnergyFpga& fpga)
{
  os << "Algo ID " << fpga.m_id << std::endl;
  os << "No. of Input Leaf Cards " << fpga.m_inputLeafCards.size() << std::endl;
  for(unsigned i=0; i < fpga.m_inputLeafCards.size(); i++)
    {
      os << (*fpga.m_inputLeafCards[i]);
    } 
  os << "Input Ex " << std::endl;
  for(unsigned i=0; i < fpga.m_inputEx.size(); i++)
    {
      os << fpga.m_inputEx[i];
    } 
  os << "Input Ey " << std::endl;
  for(unsigned i=0; i < fpga.m_inputEy.size(); i++)
    {
      os << fpga.m_inputEy[i];
    } 
  os << "Input Et " << std::endl;
  for(unsigned i=0; i < fpga.m_inputEt.size(); i++)
    {
      os << fpga.m_inputEt[i];
    } 
  os << "Output Ex " << fpga.m_outputEx << std::endl;
  os << "Output Ey " << fpga.m_outputEy << std::endl;
  os << "Output Et " << fpga.m_outputEt << std::endl;
  return os;
}

void L1GctWheelEnergyFpga::reset()
{
  for (int i=0; i<3; i++) {
    m_inputEx[i].reset();
    m_inputEy[i].reset();
    m_inputEt[i].reset();
  }
  m_outputEx.reset();
  m_outputEy.reset();
  m_outputEt.reset();
}

void L1GctWheelEnergyFpga::fetchInput()
{
  // Fetch the output values from each of our input leaf cards.
  for (int i=0; i<3; i++) {
    m_inputEx[i] = m_inputLeafCards[i]->getOutputEx();
    m_inputEy[i] = m_inputLeafCards[i]->getOutputEy();
    m_inputEt[i] = m_inputLeafCards[i]->getOutputEt();
  }
}

void L1GctWheelEnergyFpga::process()
{

  m_outputEx = m_inputEx[0] + m_inputEx[1] + m_inputEx[2];
  m_outputEy = m_inputEy[0] + m_inputEy[1] + m_inputEy[2];
  m_outputEt = m_inputEt[0] + m_inputEt[1] + m_inputEt[2];

}

///
/// assign data sources
void L1GctWheelEnergyFpga::setInputLeafCard (int i, L1GctJetLeafCard* leaf)
{
  if (i>=0 && i<3) {
    m_inputLeafCards[i] = leaf;
  }
}


///
/// set input data
void L1GctWheelEnergyFpga::setInputEnergy(int i, int ex, int ey, unsigned et)
{
  // Set the three input values from this Leaf card
  if (i>=0 && i<3) {
    m_inputEx[i].setValue(ex);
    m_inputEy[i].setValue(ey);
    m_inputEt[i].setValue(et);
  }

}

