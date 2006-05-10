#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

L1GctWheelEnergyFpga::L1GctWheelEnergyFpga() :
	m_id(id),
        m_inputLeafCards(3),
	inputEx(3),
	inputEy(3),
	inputEt(3)
{
}

L1GctWheelEnergyFpga::~L1GctWheelEnergyFpga
{
}

void L1GctWheelEnergyFpga::reset()
{
  for (int i=0; i<3; i++) {
    inputEx[i].reset();
    inputEy[i].reset();
    inputEt[i].reset();
  }
  outputEx.reset();
  outputEy.reset();
  outputEt.reset();
}

void L1GctWheelEnergyFpga::fetchInput()
{
  int ex, ey;
  unsigned et;

  unsigned long exVal, eyVal, etVal;

  int temp;
  bool ofl;

  // Fetch the output values from each of our input leaf cards.
  // Use setInputEnergy() to fill the inputEx[i] variables.
  // Assumes the number of bits used in the internal bitset<>
  // variables is the same, and the most signficant bit is
  // used for overflow.
  for (int i=0; i<3; i++) {
    exVal = m_inputLeafCards[i]->getOutputEx();
    ofl = (((int) exVal) >= Emax);
    temp = ((int) exVal) % Emax;
    if (temp>signedEmax) {
      temp -= Emax;
      if (ofl) {temp -= Emax;}
    } else {
      if (ofl) {temp += Emax;}
    }
    ex = temp;

    eyVal = m_inputLeafCards[i]->getOutputEy();
    ofl = (((int) eyVal) >= Emax);
    temp = ((int) eyVal) % Emax;
    if (temp>signedEmax) {
      temp -= Emax;
      if (ofl) {temp -= Emax;}
    } else {
      if (ofl) {temp += Emax;}
    }
    ey = temp;

    etVal = m_inputLeafCards[i]->getOutputEt();
    et = (unsigned) etVal;

    setInputEnergy(i, ex, ey, et);
  }
}

void L1GctWheelEnergyFpga::process()
{
  vector<int> exVal(3), eyVal(3), etVal(3);
  unsigned long exSum, eySum, etSum;
  bool exOfl, eyOfl, etOfl;

  int temp;

  exOfl = false;
  eyOfl = false;
  etOfl = false;

  for (int i=0; i<3; i++) {
    // Decode input Ex value with overflow bit
    temp = inputEx[i].to_ulong();
    if (temp>=Emax) {
      exOfl = true;
      temp = temp % Emax;
    }
    if (temp>=signedEmax) {
      temp -= Emax;
    }
    exVal[i] = temp;

    // Decode input Ey value with overflow bit
    temp = (int) inputEy[i].to_ulong();
    if (temp>=Emax) {
      eyOfl = true;
      temp = temp % Emax;
    }
    if (temp>=signedEmax) {
      temp -= Emax;
    }
    eyVal[i] = temp;

    // Decode input Et value with overflow bit
    temp = (int) inputEt[i].to_ulong();
    if (temp>=Emax) {
      etOfl = true;
      temp = temp % Emax;
    }
    etVal[i] = temp;
  }

  //Form Ex sum taking care of overflows
  temp = exVal[0] + exVal[1] + exVal[2];
  if (temp>=signedEmax) {
    exOfl = true;
    temp -= Emax;
  } else if (temp<-signedEmax) {
    exOfl = true;
    temp += Emax;
  }
  if (temp<0) {
    temp += Emax;
  }
  exSum = (unsigned long) temp;

  //Form Ey sum taking care of overflows
  temp = eyVal[0] + eyVal[1] + eyVal[2];
  if (temp>=signedEmax) {
    eyOfl = true;
    temp -= Emax;
  } else if (temp<-signedEmax) {
    eyOfl = true;
    temp += Emax;
  }
  if (temp<0) {
    temp += Emax;
  }
  eySum = (unsigned long) temp;

  //Form Et sum taking care of overflows
  temp = etVal[0] + etVal[1] + etVal[2];
  if (temp>=Emax) {
    etOfl = true;
    temp -= Emax;
  }
  etSum = (unsigned long) temp;

  //Convert outputs back to bitset format 
  bitset<NUM_BITS_ENERGY_DATA> exBits(exSum);
  if (exOfl) {exBits.set(OVERFLOW_BIT);}
  bitset<NUM_BITS_ENERGY_DATA> eyBits(eySum);
  if (eyOfl) {eyBits.set(OVERFLOW_BIT);}
  bitset<NUM_BITS_ENERGY_DATA> etBits(etSum);
  if (etOfl) {etBits.set(OVERFLOW_BIT);}

  outputEx = exBits;
  outputEy = eyBits;
  outputEt = etBits;

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
  unsigned long exVal, eyVal, etVal;
  bool exOfl, eyOfl, etOfl;

  int temp;

  // Transform the input variables into the correct range,
  // and set the overflow bit if they are out of range.
  // The correct range is between -signedEmax to (signedEmax-1)
  // for integer variables, 0 to (Emax-1) for unsigned.
  // Then copy all three inputs into unsigned long values
  // in the range 0 to (Emax-1).
  if (i>=0 && i<3) {
    temp = ex;
    while (temp>=signedEmax) {temp -= Emax;}
    while (temp<-signedEmax) {temp += Emax;}
    exOfl = (temp != ex);
    if (ex<0) {temp += Emax;}
    exVal = (unsigned long) temp;

    temp = ey;
    while (temp>=signedEmax) {temp -= Emax;}
    while (temp<-signedEmax) {temp += Emax;}
    eyOfl = (temp != ey);
    if (ey<0) {temp += Emax;}
    eyVal = (unsigned long) temp;

    temp = ((int) et) % Emax;
    etOfl = (temp != ((int) et));
    etVal = (unsigned long) temp;

    bitset<NUM_BITS_ENERGY_DATA> exBits(exVal);
    if (exOfl) {exBits.set(OVERFLOW_BIT);}
    bitset<NUM_BITS_ENERGY_DATA> eyBits(eyVal);
    if (eyOfl) {eyBits.set(OVERFLOW_BIT);}
    bitset<NUM_BITS_ENERGY_DATA> etBits(etVal);
    if (etOfl) {etBits.set(OVERFLOW_BIT);}

    inputEx[i] = exBits;
    inputEy[i] = eyBits;
    inputEt[i] = etBits;
  }
}
