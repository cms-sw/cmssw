#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>
using namespace std;

//DEFINE STATICS
const unsigned int L1GctJetFinderBase::MAX_JETS_OUT = 6;
const unsigned int L1GctJetFinderBase::COL_OFFSET = ((L1CaloRegionDetId::N_ETA)/2)+1;
const unsigned int L1GctJetFinderBase::N_JF_PER_WHEEL = ((L1CaloRegionDetId::N_PHI)/2);

const unsigned int L1GctJetFinderBase::MAX_REGIONS_IN = L1GctJetFinderBase::COL_OFFSET*L1GctJetFinderBase::N_COLS;
const unsigned int L1GctJetFinderBase::N_COLS = 2;
const unsigned int L1GctJetFinderBase::CENTRAL_COL0 = 0;


L1GctJetFinderBase::L1GctJetFinderBase(int id):
  m_id(id),
  m_neighbourJetFinders(2),
  m_gotNeighbourPointers(false),
  m_jetEtCalLut(0),
  m_inputRegions(MAX_REGIONS_IN),
  m_outputJets(MAX_JETS_OUT)
{
  // Call reset to initialise vectors for input and output
  this->reset();
  //Check jetfinder setup
  if(m_id < 0 || m_id >= static_cast<int>(L1CaloRegionDetId::N_PHI))
  {
    throw cms::Exception("L1GctSetupError")
    << "L1GctJetFinderBase::L1GctJetFinderBase() : Jet Finder ID " << m_id << " has been incorrectly constructed!\n"
    << "ID number should be between the range of 0 to " << L1CaloRegionDetId::N_PHI-1 << "\n";
  } 
  // Initialise parameters for Region input calculations
  static const unsigned NPHI = L1CaloRegionDetId::N_PHI;
  m_minColThisJf = (NPHI + m_id*2 - this->centralCol0()) % NPHI;
}

L1GctJetFinderBase::~L1GctJetFinderBase()
{
}

/// Set pointers to neighbours
void L1GctJetFinderBase::setNeighbourJetFinders(std::vector<L1GctJetFinderBase*> neighbours)
{
  if (neighbours.size()==2) {
    m_neighbourJetFinders = neighbours;
  } else {
    throw cms::Exception("L1GctSetupError")
      << "L1GctJetFinderBase::setNeighbourJetFinders() : In Jet Finder ID " << m_id 
      << " size of input vector should be 2, but is in fact " << neighbours.size() << "\n";
  }
  if (m_neighbourJetFinders.at(0) == 0) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctJetFinderBase::setNeighbourJetFinders() : In Jet Finder ID " << m_id 
      << " first neighbour pointer is set to zero\n";
  }
  if (m_neighbourJetFinders.at(1) == 0) {
    throw cms::Exception("L1GctSetupError")
      << "L1GctJetFinderBase::setNeighbourJetFinders() : In Jet Finder ID " << m_id 
      << " second neighbour pointer is set to zero\n";
  }
  m_gotNeighbourPointers = true;
}

/// Set pointer to calibration Lut - needed to complete the setup
void L1GctJetFinderBase::setJetEtCalibrationLut(L1GctJetEtCalibrationLut* lut)
{
  m_jetEtCalLut = lut;
}

ostream& operator << (ostream& os, const L1GctJetFinderBase& algo)
{
  os << "ID = " << algo.m_id << endl;
  os << "JetEtCalibrationLut* = " <<  algo.m_jetEtCalLut << endl;
  os << "No of input regions " << algo.m_inputRegions.size() << endl;
//   for(unsigned i=0; i < algo.m_inputRegions.size(); ++i)
//     {
//       os << algo.m_inputRegions.at(i); 
//     }
  os << "No of output jets " << algo.m_outputJets.size() << endl;
//   for(unsigned i=0; i < algo.m_outputJets.size(); ++i)
//     {
//       os << algo.m_outputJets.at(i); 
//     }
  os << "Output Et strip 0 " << algo.m_outputEtStrip0 << endl;
  os << "Output Et strip 1 " << algo.m_outputEtStrip1 << endl;
  os << "Output Ht " << algo.m_outputHt << endl;
  os << endl;

  return os;
}


void L1GctJetFinderBase::reset()
{
  m_inputRegions.clear();
  m_inputRegions.resize(this->maxRegionsIn());
  m_outputJets.clear();
  m_outputJets.resize(MAX_JETS_OUT);
  m_sortedJets.clear();
  m_sortedJets.resize(MAX_JETS_OUT);
  
  m_sentProtoJets.clear();
  m_sentProtoJets.resize(MAX_JETS_OUT);
  m_rcvdProtoJets.clear();
  m_rcvdProtoJets.resize(MAX_JETS_OUT);
  m_keptProtoJets.clear();
  m_keptProtoJets.resize(MAX_JETS_OUT);

  m_outputEtStrip0 = 0;
  m_outputEtStrip1 = 0;
  m_outputHt = 0;
}

// This is how the regions from the RCT get into the GCT for processing 
void L1GctJetFinderBase::setInputRegion(L1CaloRegion region)
{
  static const unsigned NPHI = L1CaloRegionDetId::N_PHI;
  unsigned crate = region.rctCrate();
  // Find the column for this region in a global (eta,phi) array
  // Note the column numbers here are not the same as region->gctPhi()
  // because the RCT crates are not numbered from phi=0.
  unsigned colAbsolute = crate*2 + region.rctPhi();
  unsigned colRelative = ((colAbsolute+NPHI) - m_minColThisJf) % NPHI;
  if (colRelative < this->nCols()) {
    // We are in the right range in phi
    // Now check we are in the right wheel (positive or negative eta)
    if ( (crate/N_JF_PER_WHEEL) == (m_id/N_JF_PER_WHEEL) ) {
      unsigned i = colRelative*COL_OFFSET + region.rctEta() + 1;
      m_inputRegions.at(i) = region;
    } else {
      // Accept neighbouring regions from the other wheel
      if (region.rctEta() == 0) {
	unsigned i = colRelative*COL_OFFSET;
	m_inputRegions.at(i) = region;
      }
    }
  }
}

// PROTECTED METHODS BELOW
/// fetch the protoJets from neighbour jetFinder
void L1GctJetFinderBase::fetchProtoJetsFromNeighbour(const fetchType ft)
{
  switch (ft) {
  case TOP : 
    m_rcvdProtoJets = m_neighbourJetFinders.at(0)->getSentProtoJets(); break;
  case BOT :
    m_rcvdProtoJets = m_neighbourJetFinders.at(1)->getSentProtoJets(); break;
  case TOPBOT :
    // Copy half the jets from each neighbour
    static const unsigned int MAX_TOPBOT_JETS = MAX_JETS_OUT/2;
    unsigned j=0;
    RegionsVector temp;
    temp = m_neighbourJetFinders.at(0)->getSentProtoJets();
    for ( ; j<MAX_TOPBOT_JETS; ++j) {
      m_rcvdProtoJets.at(j) = temp.at(j);
    } 
    temp = m_neighbourJetFinders.at(1)->getSentProtoJets();
    for ( ; j<MAX_JETS_OUT; ++j) {
      m_rcvdProtoJets.at(j) = temp.at(j);
    }     
    break;
  }
}


/// Sort the found jets. All jetFinders should call this in process().
void L1GctJetFinderBase::sortJets()
{
  //transform the jets to the final GCT output format
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    m_sortedJets.at(j) = m_outputJets.at(j).jetCand(m_jetEtCalLut);
  }
  //presort the jets into descending order of energy
  sort(m_sortedJets.begin(), m_sortedJets.end(), rankGreaterThan());
}
   
/// Fill the Et strip sums and Ht sum. All jetFinders should call this in process().
void L1GctJetFinderBase::doEnergySums()
{
  //calculate the raw Et strip sums
  m_outputEtStrip0 = calcEtStrip(0);
  m_outputEtStrip1 = calcEtStrip(1);

  //calculate the Ht
  m_outputHt = calcHt();
    
  return;
}


// Calculates total (raw) energy in a phi strip
L1GctUnsignedInt<12> L1GctJetFinderBase::calcEtStrip(const UShort strip) const
{
  if (strip !=0 && strip != 1) {
    throw cms::Exception("L1GctProcessingError")
      << "L1GctJetFinderBase::calcEtStrip() has been called with strip number "
      << strip << "; should be 0 or 1 \n";
  } 
  // Add the Et values from regions 13 to 23 for strip 0,
  //     the Et values from regions 25 to 35 for strip 1.
  unsigned et = 0;
  bool of = false;
  unsigned offset = COL_OFFSET * (strip+centralCol0());
  for (UShort i=1; i < COL_OFFSET; ++i) {
    offset++;
    et += m_inputRegions.at(offset).et();
    of |= m_inputRegions.at(offset).overFlow();
  }
  L1GctUnsignedInt<12> temp(et);
  temp.setOverFlow(temp.overFlow() || of);
  return temp;
}

// Calculates total calibrated energy in jets (Ht) sum
L1GctUnsignedInt<12> L1GctJetFinderBase::calcHt() const
{    
  unsigned ht = 0;
  bool of = false;
  for(UShort i=0; i < MAX_JETS_OUT; ++i)
  {
    // Only sum Ht for valid jets
    if (!m_outputJets.at(i).isNullJet()) {
      ht += m_outputJets.at(i).calibratedEt(m_jetEtCalLut);
      of |= m_outputJets.at(i).overFlow();
    }
  }
  L1GctUnsignedInt<12> temp(ht);
  temp.setOverFlow(temp.overFlow() || of);
  return temp;
}
