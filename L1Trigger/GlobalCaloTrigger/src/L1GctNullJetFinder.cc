#include "L1Trigger/GlobalCaloTrigger/interface/L1GctNullJetFinder.h"

//DEFINE STATICS
const unsigned int L1GctNullJetFinder::MAX_REGIONS_IN = (((L1CaloRegionDetId::N_ETA)/2)+1)*L1GctNullJetFinder::N_COLS;
const unsigned int L1GctNullJetFinder::N_COLS = 4;
const unsigned int L1GctNullJetFinder::CENTRAL_COL0 = 1;

L1GctNullJetFinder::L1GctNullJetFinder(int id):
  L1GctJetFinderBase(id)
{
  this->reset();
}

L1GctNullJetFinder::~L1GctNullJetFinder()
{
}

std::ostream& operator << (std::ostream& os, const L1GctNullJetFinder& algo)
{
  os << "===L1GctNullJetFinder===" << std::endl;
  const L1GctJetFinderBase* temp = &algo;
  os << *temp;
  return os;
}

void L1GctNullJetFinder::fetchInput()
{
  // Get rid of any input objects that may have been stored (!)
  resetProcessor();
  setupObjects();
}

void L1GctNullJetFinder::process() 
{
  if (setupOk()) {
    // NO jet finder so all jets (and intermediate clusters etc) will be null
    // as created by the call to setupObjects() above
    doEnergySums();
  }
}
