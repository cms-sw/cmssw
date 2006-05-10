#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

L1GctWheelJetFpga::L1GctWheelJetFpga(int id) :
  m_id(id),
  m_inputLeafCards(3),
  inputJets(54),
  inputHt(3),
  outputJets(12),
  outputJc(12)
{
}

L1GctWheelJetFpga::~L1GctWheelJetFpga()
{
}

void L1GctWheelJetFpga::reset()
{
  for (int i=0; i<3; i++) {
    inputHt[i].reset();
  }
  outputHt.reset();
  for (int i=0; i<12; i++) {
    outputJc[i].reset();
  }
}

void L1GctWheelJetFpga::fetchInput()
{
  unsigned ht;

  unsigned long htVal;

  for (int i=0; i<3; i++) {

    // Deal with the jets

    // Deal with the Ht inputs
    // Fetch the output values from each of our input leaf cards.
    // Use setInputHt() to fill the inputEx[i] variables.
    htVal = m_inputLeafCards[i]->getOutputHt();
    ht = (unsigned) htVal;

    setInputHt(i, ht);
  }
}

void L1GctWheelJetFpga::process()
{
  vector<int> htVal(3);
  unsigned long htSum;
  bool htOfl;

  int temp;

  // Deal with the jets

  // Deal with the jet counts

  // Deal with the Ht summing
  // Form the Ht sum from the inputs
  // sent from the Leaf cards
  htOfl = false;

  for (int i=0; i<3; i++) {

    // Decode input Ht value with overflow bit
    temp = (int) inputHt[i].to_ulong();
    if (temp>=Emax) {
      htOfl = true;
      temp = temp % Emax;
    }
    htVal[i] = temp;
  }

  // Form Et sum taking care of overflows
  temp = htVal[0] + htVal[1] + htVal[2];
  if (temp>=Emax) {
    htOfl = true;
    temp -= Emax;
  }
  htSum = (unsigned long) temp;

  // Convert output back to bitset format 
  bitset<NUM_BITS_ENERGY_DATA> htBits(htSum);
  if (htOfl) {htBits.set(OVERFLOW_BIT);}

  outputHt = htBits;

}

void L1GctWheelJetFpga::setInputLeafCard (int i, L1GctJetLeafCard* leaf)
{
  if (i>=0 && i<3) {
    m_inputLeafCards[i] = leaf;
  }
}

void L1GctWheelJetFpga::setInputJet(int i, L1GctJet jet)
{
}

void L1GctWheelJetFpga::setInputHt (int i, unsigned ht)
{	
  unsigned long htVal;
  bool htOfl;

  int temp;

  if (i>=0 && i<3) {
    // Transform the input variables into the correct range,
    // and set the overflow bit if they are out of range.
    // The correct range is between 0 to (Emax-1).
    // Then copy the inputs into an unsigned long variable.
    temp = ((int) ht) % Emax;
    htOfl = (temp != ((int) ht));
    htVal = (unsigned long) temp;

    // Copy the data into the internal bitset format.
    bitset<NUM_BITS_ENERGY_DATA> htBits(htVal);
    if (htOfl) {htBits.set(OVERFLOW_BIT);}

    inputHt[i] = htBits;
  }
}
