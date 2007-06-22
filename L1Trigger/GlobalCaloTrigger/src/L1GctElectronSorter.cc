#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"

#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>

using namespace std;


L1GctElectronSorter::L1GctElectronSorter(int nInputs, bool iso):
  m_id(nInputs),
  m_isolation(iso),
  m_inputCands(nInputs*4),
  m_outputCands(4)
{}  

L1GctElectronSorter::~L1GctElectronSorter()
{
}

// clear buffers
void L1GctElectronSorter::reset() {
  m_inputCands.clear();
  m_inputCands.resize(m_id*4);
  m_outputCands.clear();
  m_outputCands.resize(4);
}

// get the input data
void L1GctElectronSorter::fetchInput() {
  // This does nothing, assume the input candidates get pushed in
}

//Process sorts the electron candidates after rank and stores the highest four (in the outputCands vector)
void L1GctElectronSorter::process() {

  //Convert from caloEmCand to gctEmCand and make temporary copy of data
  std::vector<L1GctEmCand> data(m_inputCands.size());
  for (unsigned i=0; i<m_inputCands.size(); i++) {
    data.at(i) = L1GctEmCand(m_inputCands.at(i));
  }

  //Then sort it
  sort(data.begin(),data.end(),rank_gt());

  //Copy data to output buffer
  for(int i = 0; i<4; i++){
    m_outputCands.at(i) = data.at(i);
  }
}

void L1GctElectronSorter::setInputEmCand(L1CaloEmCand cand){
  unsigned crate = cand.rctCrate();
  unsigned input = ((crate%9 < 4) ? (crate%9) : (crate%9 - 4));
  unsigned i = input*4;
  for (unsigned j=0; j<4; j++) {
    if (m_inputCands.at(i).rank() == 0) {
      m_inputCands.at(i) = cand;
      break;
    }
    i++;
  }
}

std::ostream& operator<<(std::ostream& s, const L1GctElectronSorter& ems) {
  s << "===L1GctElectronSorter===" << std::endl;
  s << "Algo type = " << ems.m_isolation << std::endl;
  s << "No of Electron Input Candidates = " << ems.m_inputCands.size()<< std::endl;
  s << "No of Electron Output Candidates = " << ems.m_outputCands.size()<< std::endl;
  return s;
}

