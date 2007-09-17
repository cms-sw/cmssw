
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"

const unsigned int L1GctJetCounterSetup::MAX_CUT_TYPE = nullCutType;
const unsigned int L1GctJetCounterSetup::MAX_JET_COUNTERS = 12;

L1GctJetCounterSetup::L1GctJetCounterSetup(const cutsListForWheelCard& cuts) :
  m_jetCounterCuts(cuts)
{
}

L1GctJetCounterSetup::L1GctJetCounterSetup() :
  m_jetCounterCuts()
{
}

L1GctJetCounterSetup::~L1GctJetCounterSetup() {}

L1GctJetCounterSetup::cutsListForJetCounter
L1GctJetCounterSetup::getCutsForJetCounter(unsigned i) const
{
  cutsListForJetCounter result;
  if (i<numberOfJetCounters()) {
    result = m_jetCounterCuts.at(i);
  }
  return result;
}

void L1GctJetCounterSetup::addJetCounter(const L1GctJetCounterSetup::cutsListForJetCounter& cuts)
{
  m_jetCounterCuts.push_back(cuts);
}
