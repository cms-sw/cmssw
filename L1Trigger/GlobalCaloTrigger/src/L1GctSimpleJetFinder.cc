#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSimpleJetFinder.h"
 
#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>
using namespace std;

//DEFINE STATICS
const unsigned int L1GctSimpleJetFinder::MAX_REGIONS_IN = (((L1GctMap::N_RGN_ETA)/2)+1)*L1GctSimpleJetFinder::N_COLS;

const int L1GctSimpleJetFinder::N_COLS = 2;
const unsigned int L1GctSimpleJetFinder::CENTRAL_COL0 = 0;

L1GctSimpleJetFinder::L1GctSimpleJetFinder(int id, vector<L1GctSourceCard*> sourceCards,
					   L1GctJetEtCalibrationLut* jetEtCalLut):
  L1GctJetFinderBase(id, sourceCards, jetEtCalLut)
{
  this->reset();
}

L1GctSimpleJetFinder::~L1GctSimpleJetFinder()
{
}

ostream& operator << (ostream& os, const L1GctSimpleJetFinder& algo)
{
  os << "===L1GctSimpleJetFinder===" << endl;
  const L1GctJetFinderBase* temp = &algo;
  os << *temp;
  return os;
}

void L1GctSimpleJetFinder::fetchInput()
{
  fetchCentreStripsInput();
}

void L1GctSimpleJetFinder::process() 
{
  findJets();
  sortJets();
  doEnergySums();
}

void L1GctSimpleJetFinder::findJets()
{
  // Here's a dummy jetfinder, currently does nothing
  UShort jetNum = 0; //holds the number of jets currently found
  for(UShort column = 1; column <=2; ++column)  //Find jets in the central search region
  {
    //don't include row zero as it is not in the search region
    for (UShort row = 1; row < COL_OFFSET; ++row)  
    {
      if (false) {
        assert(jetNum < MAX_JETS_OUT);
                
        m_outputJets.at(jetNum).setRawsum(0);
	// Use the jetFinder m_id to assign the eta and phi in global coordinates here
        m_outputJets.at(jetNum).setEta(map->globalEta((row-1),   (m_id/N_JF_PER_WHEEL)));
        m_outputJets.at(jetNum).setPhi(map->globalPhi((column-1),(m_id%N_JF_PER_WHEEL)));
        ++jetNum;
      }
    }
  }
}

