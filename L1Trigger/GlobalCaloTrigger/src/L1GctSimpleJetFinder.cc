#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSimpleJetFinder.h"
 
#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>
using namespace std;

//DEFINE STATICS
const unsigned int L1GctSimpleJetFinder::MAX_REGIONS_IN = (((L1CaloRegionDetId::N_ETA)/2)+1)*L1GctSimpleJetFinder::N_COLS;

const int L1GctSimpleJetFinder::N_COLS = 2;
const unsigned int L1GctSimpleJetFinder::CENTRAL_COL0 = 0;

L1GctSimpleJetFinder::L1GctSimpleJetFinder(int id, vector<L1GctSourceCard*> sourceCards):
  L1GctJetFinderBase(id, sourceCards)
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

/// HERE IS THE JETFINDER CODE

void L1GctSimpleJetFinder::findJets()
{
  // Here's a dummy jetfinder, currently does nothing
  UShort jetNum = 0; //holds the number of jets currently found
  UShort centreIndex = COL_OFFSET*this->centralCol0();
  for(UShort column = 0; column <2; ++column)  //Find jets in the central search region
  {
    //don't include row zero as it is not in the search region
    ++centreIndex;
    for (UShort row = 1; row < COL_OFFSET; ++row)  
    {
      if (false) {
        assert(jetNum < MAX_JETS_OUT);
                
        m_outputJets.at(jetNum).setRawsum(m_inputRegions.at(centreIndex).et());
        m_outputJets.at(jetNum).setDetId(m_inputRegions.at(centreIndex).id());
        ++jetNum;
      }
      ++centreIndex;
    }
  }
}

