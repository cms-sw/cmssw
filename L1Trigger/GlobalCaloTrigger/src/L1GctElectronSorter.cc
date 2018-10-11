#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include <algorithm>

L1GctElectronSorter::L1GctElectronSorter(int nInputs, bool iso):
  L1GctProcessor(),
  m_id(nInputs),
  m_isolation(iso),
  m_inputCands(nInputs*4),
  m_outputCands(4)
{}  

L1GctElectronSorter::~L1GctElectronSorter()
{
}

// clear buffers
void L1GctElectronSorter::resetProcessor() {
  m_inputCands.clear();
  m_inputCands.resize(m_id*4);

  m_outputCands.clear();
  m_outputCands.resize(4);
}

/// Initialise inputs with null objects for the correct bunch crossing
/// If no other input candidates "arrive", we have the correct
/// bunch crossing to propagate through the processing.
void L1GctElectronSorter::setupObjects() {
  /// Create a null input electron with the right bunch crossing, 
  /// and fill the input candidates with copies of this.
  L1CaloEmCand temp;
  temp.setBx(bxAbs());
  m_inputCands.assign(m_id*4, temp);
}

// get the input data
void L1GctElectronSorter::fetchInput() {
  // This does nothing, assume the input candidates get pushed in
}

//Process sorts the electron candidates after rank and stores the highest four (in the outputCands vector)
void L1GctElectronSorter::process() {

  //Convert from caloEmCand to gctEmCand and make temporary copy of data
  std::vector<prioritisedEmCand> data(m_inputCands.size());
  // Assign a "priority" for sorting - this assumes the candidates
  // have already been filled in "priority order"
  for (unsigned i=0; i<m_inputCands.size(); i++) {
    prioritisedEmCand c(m_inputCands.at(i), i);
    data.at(i) = c;
  }

  //Then sort it
  sort(data.begin(),data.end(),rankByGt);

  //Copy data to output buffer
  for(int i = 0; i<4; i++){
    m_outputCands.at(i) = data.at(i).emCand;
  }
}

void L1GctElectronSorter::setInputEmCand(const L1CaloEmCand& cand){
  // Fills the candidates in "priority order"
  // The lowest numbered RCT crate in each FPGA has highest priority.
  // We distinguish the two FPGAs on a leaf card by the number of inputs.
  // FPGA U1 has 5 inputs (crates 4-8) and FPGA U2 has 4 inputs (crates 0-3).
  // Within a crate the four input candidates are arranged in the order
  // that they arrive on the cable, using the index() method.
  unsigned crate = cand.rctCrate();
  unsigned input = ( (m_id==4) ? (crate%9) : (crate%9 - 4) );
  unsigned i = input*4 + (3-cand.index());
  if (m_inputCands.at(i).rank()==0) {
    m_inputCands.at(i) = cand;
  }
}

std::ostream& operator<<(std::ostream& s, const L1GctElectronSorter& ems) {
  s << "===L1GctElectronSorter===" << std::endl;
  s << "Algo type = " << ems.m_isolation << std::endl;
  s << "No of Electron Input Candidates = " << ems.m_inputCands.size()<< std::endl;
  s << "No of Electron Output Candidates = " << ems.m_outputCands.size()<< std::endl;
  return s;
}

