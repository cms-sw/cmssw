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
    m_inputEx[i] = m_inputLeafCards[i]->outputEx();
    m_inputEy[i] = m_inputLeafCards[i]->outputEy();
    m_inputEt[i] = m_inputLeafCards[i]->outputEt();
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

