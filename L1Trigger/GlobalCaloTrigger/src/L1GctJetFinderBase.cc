#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternJetData.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DEFINE STATICS
const unsigned int L1GctJetFinderBase::MAX_JETS_OUT = 6;
const unsigned int L1GctJetFinderBase::N_EXTRA_REGIONS_ETA00 = 2;
const unsigned int L1GctJetFinderBase::COL_OFFSET = L1GctJetFinderParams::NUMBER_ETA_VALUES+N_EXTRA_REGIONS_ETA00;
const unsigned int L1GctJetFinderBase::N_JF_PER_WHEEL = ((L1CaloRegionDetId::N_PHI)/2);

const unsigned int L1GctJetFinderBase::MAX_REGIONS_IN = L1GctJetFinderBase::COL_OFFSET*L1GctJetFinderBase::N_COLS;
const unsigned int L1GctJetFinderBase::N_COLS = 2;
const unsigned int L1GctJetFinderBase::CENTRAL_COL0 = 0;


L1GctJetFinderBase::L1GctJetFinderBase(int id):
  L1GctProcessor(),
  m_id(id),
  m_neighbourJetFinders(2),
  m_idInRange(false),
  m_gotNeighbourPointers(false),
  m_gotJetFinderParams(false),
  m_gotJetEtCalLuts(false),
  m_gotChannelMask(false),
  m_positiveEtaWheel(id >= (int) (L1CaloRegionDetId::N_PHI/2)), m_minColThisJf(0),
  m_CenJetSeed(0), m_FwdJetSeed(0), m_TauJetSeed(0), m_EtaBoundry(0),
  m_jetEtCalLuts(),
  m_useImprovedTauAlgo(false), m_ignoreTauVetoBitsForIsolation(false), m_tauIsolationThreshold(0),
  m_HttSumJetThreshold(0), m_HtmSumJetThreshold(0),
  m_EttMask(), m_EtmMask(), m_HttMask(), m_HtmMask(), 
  m_inputRegions(MAX_REGIONS_IN),
  m_sentProtoJets(MAX_JETS_OUT), m_rcvdProtoJets(MAX_JETS_OUT), m_keptProtoJets(MAX_JETS_OUT),
  m_outputJets(MAX_JETS_OUT), m_sortedJets(MAX_JETS_OUT),
  m_outputHfSums(),
  m_outputJetsPipe(MAX_JETS_OUT),
  m_outputEtSumPipe(), m_outputExSumPipe(), m_outputEySumPipe(), 
  m_outputHtSumPipe(), m_outputHxSumPipe(), m_outputHySumPipe()
{
  // Call reset to initialise vectors for input and output
  this->reset();
  //Check jetfinder setup
  if(m_id < 0 || m_id >= static_cast<int>(L1CaloRegionDetId::N_PHI))
  {
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	<< "L1GctJetFinderBase::L1GctJetFinderBase() : Jet Finder ID " << m_id << " has been incorrectly constructed!\n"
	<< "ID number should be between the range of 0 to " << L1CaloRegionDetId::N_PHI-1 << "\n";
    } 
  } else { m_idInRange = true; }

}

L1GctJetFinderBase::~L1GctJetFinderBase()
{
}

/// Set pointers to neighbours
void L1GctJetFinderBase::setNeighbourJetFinders(const std::vector<L1GctJetFinderBase*>& neighbours)
{
  m_gotNeighbourPointers = true;
  if (neighbours.size()==2) {
    m_neighbourJetFinders = neighbours;
  } else {
    m_gotNeighbourPointers = false;
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	  << "L1GctJetFinderBase::setNeighbourJetFinders() : In Jet Finder ID " << m_id 
	  << " size of input vector should be 2, but is in fact " << neighbours.size() << "\n";
    }
  }
  if (m_neighbourJetFinders.at(0) == 0) {
    m_gotNeighbourPointers = false;
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	  << "L1GctJetFinderBase::setNeighbourJetFinders() : In Jet Finder ID " << m_id 
	  << " first neighbour pointer is set to zero\n";
    }
  }
  if (m_neighbourJetFinders.at(1) == 0) {
    m_gotNeighbourPointers = false;
    if (m_verbose) {
      edm::LogWarning("L1GctSetupError")
	  << "L1GctJetFinderBase::setNeighbourJetFinders() : In Jet Finder ID " << m_id 
	  << " second neighbour pointer is set to zero\n";
    }
  }
  if (!m_gotNeighbourPointers && m_verbose) {
    edm::LogError("L1GctSetupError") << "Jet Finder ID " << m_id << " has incorrect assignment of neighbour pointers";
  }
}

/// Set pointer to parameters - needed to complete the setup
void L1GctJetFinderBase::setJetFinderParams(const L1GctJetFinderParams* jfpars)
{
  m_CenJetSeed = jfpars->getCenJetEtSeedGct();
  m_FwdJetSeed = jfpars->getForJetEtSeedGct();
  m_TauJetSeed = jfpars->getTauJetEtSeedGct();
  m_EtaBoundry = jfpars->getCenForJetEtaBoundary();
  m_tauIsolationThreshold = jfpars->getTauIsoEtThresholdGct();
  m_HttSumJetThreshold    = jfpars->getHtJetEtThresholdGct();
  m_HtmSumJetThreshold    = jfpars->getMHtJetEtThresholdGct();
  m_gotJetFinderParams = true;
}

/// Set pointer to calibration Lut - needed to complete the setup
void L1GctJetFinderBase::setJetEtCalibrationLuts(const L1GctJetFinderBase::lutPtrVector& jfluts)
{
  m_jetEtCalLuts = jfluts;
  m_gotJetEtCalLuts = (jfluts.size() >= L1GctJetFinderParams::NUMBER_ETA_VALUES);
}

/// Set et sum masks from ChannelMask object - needed to complete the setup
void L1GctJetFinderBase::setEnergySumMasks(const L1GctChannelMask* chmask)
{
  bool matchCheckEttAndEtm = true;
  if (chmask != 0) {
    static const unsigned N_ETA = L1GctJetFinderParams::NUMBER_ETA_VALUES;
    for (unsigned ieta=0; ieta<N_ETA; ++ieta) {
      unsigned globalEta = (m_positiveEtaWheel ? N_ETA+ieta : N_ETA - (ieta+1) );
      m_EttMask[ieta] = chmask->totalEtMask(globalEta);
      m_EtmMask[ieta] = chmask->missingEtMask(globalEta);
      m_HttMask[ieta] = chmask->totalHtMask(globalEta);
      m_HtmMask[ieta] = chmask->missingHtMask(globalEta);

      matchCheckEttAndEtm &= (m_EttMask[ieta] == m_EtmMask[ieta]);
    }
    if (!matchCheckEttAndEtm)
      edm::LogWarning("L1GctSetupError") 
	<< "L1GctJetFinderBase::setEnergySumMasks() : In Jet Finder ID " << m_id 
	<< " setting eta-dependent masks for Et sums: you cannot have different masks for total and missing Et\n";
    m_gotChannelMask = true;
  }
}


std::ostream& operator << (std::ostream& os, const L1GctJetFinderBase& algo)
{
  using std::endl;
  os << "ID = " << algo.m_id << endl;
  os << "Calibration lut pointers stored for " << algo.m_jetEtCalLuts.size() << " eta bins" << endl;
  for (unsigned ieta=0; ieta<algo.m_jetEtCalLuts.size(); ieta++) {
    os << "Eta bin " << ieta << ", JetEtCalibrationLut* = " <<  algo.m_jetEtCalLuts.at(ieta) << endl;
  }
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
  os << "Output total scalar Et " << algo.m_outputEtSum << endl;
  os << "Output vector Et x component " << algo.m_outputExSum << endl;
  os << "Output vector Et y component " << algo.m_outputEySum << endl;
  os << "Output total scalar Ht " << algo.m_outputHtSum << endl;
  os << "Output vector Ht x component " << algo.m_outputHxSum << endl;
  os << "Output vector Ht y component " << algo.m_outputHySum << endl;
  os << endl;

  return os;
}


void L1GctJetFinderBase::resetProcessor()
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

  m_outputEtSum = 0;
  m_outputExSum = 0;
  m_outputEySum = 0;
  m_outputHtSum = 0;
  m_outputHxSum = 0;
  m_outputHySum = 0;

  m_outputHfSums.reset();
}

void L1GctJetFinderBase::resetPipelines()
{
  m_outputJetsPipe.reset(numOfBx());
  m_outputEtSumPipe.reset(numOfBx());
  m_outputExSumPipe.reset(numOfBx());
  m_outputEySumPipe.reset(numOfBx());
  m_outputHtSumPipe.reset(numOfBx());
  m_outputHxSumPipe.reset(numOfBx());
  m_outputHySumPipe.reset(numOfBx());
}

/// Initialise inputs with null objects for the correct bunch crossing
/// If no other input candidates "arrive", we have the correct
/// bunch crossing to propagate through the processing.
void L1GctJetFinderBase::setupObjects()
{
  /// Create a null input region with the right bunch crossing, 
  /// and fill the input candidates with copies of this.
  L1GctRegion tempRgn;
  tempRgn.setBx(bxAbs());
  m_inputRegions.assign(this->maxRegionsIn(), tempRgn);

  /// The same for the lists of pre-clustered jets
  /// passed between neighbour jetFinders
  m_sentProtoJets.assign(MAX_JETS_OUT, tempRgn);
  m_rcvdProtoJets.assign(MAX_JETS_OUT, tempRgn);
  m_keptProtoJets.assign(MAX_JETS_OUT, tempRgn);

  /// The same for the lists of output jets
  L1GctJet tempJet;
  tempJet.setBx(bxAbs());
  m_outputJets.assign(MAX_JETS_OUT, tempJet);
}

// This is how the regions from the RCT get into the GCT for processing 
void L1GctJetFinderBase::setInputRegion(const L1CaloRegion& region)
{
  static const unsigned NPHI = L1CaloRegionDetId::N_PHI;
  static const unsigned N_00 = N_EXTRA_REGIONS_ETA00;
  unsigned crate = region.rctCrate();
  // Find the column for this region in a global (eta,phi) array
  // Note the column numbers here are not the same as region->gctPhi()
  // because the RCT crates are not numbered from phi=0.
  unsigned colAbsolute = (crate+1)*2 + region.rctPhi();
  unsigned colRelative = ((colAbsolute+NPHI) - m_minColThisJf) % NPHI;
  if (colRelative < this->nCols()) {
    // We are in the right range in phi
    // Now check we are in the right wheel (positive or negative eta)
    if ( (crate/N_JF_PER_WHEEL) == (m_id/N_JF_PER_WHEEL) ) {
      unsigned i = colRelative*COL_OFFSET + N_00 + region.rctEta();
      m_inputRegions.at(i) = L1GctRegion::makeJfInputRegion(region);
    } else {
      // Accept neighbouring regions from the other wheel
      if (region.rctEta() < N_00) {
	unsigned i = colRelative*COL_OFFSET + N_00 - (region.rctEta()+1);
	m_inputRegions.at(i) = L1GctRegion::makeJfInputRegion(region);
      }
    }
  }
}

/// get output jets in raw format - to be stored in the event
std::vector< L1GctInternJetData > L1GctJetFinderBase::getInternalJets() const {

  std::vector< L1GctInternJetData > result;
  for (RawJetVector::const_iterator jet=m_outputJetsPipe.contents.begin();
       jet!=m_outputJetsPipe.contents.end(); jet++) {
    result.push_back( L1GctInternJetData::fromEmulator(jet->id(),
						       jet->bx(),
						       jet->calibratedEt(m_jetEtCalLuts.at(jet->rctEta())), 
						       jet->overFlow(), 
						       jet->tauVeto(),
						       jet->hwEta(),
						       jet->hwPhi(),
						       jet->rank(m_jetEtCalLuts.at(jet->rctEta())) ) );
  }
  return result;

}

/// get et sums in raw format - to be stored in the event
std::vector< L1GctInternEtSum > L1GctJetFinderBase::getInternalEtSums() const {

  std::vector< L1GctInternEtSum > result;
  for (int bx=0; bx<numOfBx(); bx++) {
    result.push_back( L1GctInternEtSum::fromEmulatorJetTotEt ( m_outputEtSumPipe.contents.at(bx).value(),
							       m_outputEtSumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorJetMissEt( m_outputExSumPipe.contents.at(bx).value(),
							       m_outputExSumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorJetMissEt( m_outputEySumPipe.contents.at(bx).value(),
							       m_outputEySumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
    result.push_back( L1GctInternEtSum::fromEmulatorJetTotHt ( m_outputHtSumPipe.contents.at(bx).value(),
							       m_outputHtSumPipe.contents.at(bx).overFlow(),
							       static_cast<int16_t> (bx-bxMin()) ) );
  }
  return result;
}

std::vector< L1GctInternHtMiss > L1GctJetFinderBase::getInternalHtMiss() const {

  std::vector< L1GctInternHtMiss > result;
  for (int bx=0; bx<numOfBx(); bx++) {
    result.push_back( L1GctInternHtMiss::emulatorJetMissHt( m_outputHxSumPipe.contents.at(bx).value(),
							    m_outputHySumPipe.contents.at(bx).value(),
							    m_outputHxSumPipe.contents.at(bx).overFlow(),
							    static_cast<int16_t> (bx-bxMin()) ) );
  }
  return result;

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
  JetVector tempJets(MAX_JETS_OUT);
  for (unsigned j=0; j<MAX_JETS_OUT; j++) {
    tempJets.at(j) = m_outputJets.at(j).jetCand(m_jetEtCalLuts);
  }

  // Sort the jets
  L1GctJetSorter jSorter(tempJets);
  m_sortedJets = jSorter.getSortedJets();

  //store jets in "pipeline memory" for checking
  m_outputJetsPipe.store(m_outputJets, bxRel());
}
   
/// Fill the Et strip sums and Ht sum. All jetFinders should call this in process().
void L1GctJetFinderBase::doEnergySums()
{

  // Refactored energy sums code - find scalar and vector sums
  // of Et and Ht instead of strip stums
  doEtSums();
  doHtSums();

  //calculate the Hf tower Et sums and tower-over-threshold counts
  m_outputHfSums = calcHfSums();
    
  return;
}

// Calculates scalar and vector sum of Et over input regions
void L1GctJetFinderBase::doEtSums() {
  unsigned et0 = 0;
  unsigned et1 = 0;
  bool of = false;

  // Add the Et values from regions  2 to 12 for strip 0,
  //     the Et values from regions 15 to 25 for strip 1.
  unsigned offset = COL_OFFSET * centralCol0();
  unsigned ieta = 0;
  for (UShort i=offset+N_EXTRA_REGIONS_ETA00; i < offset+COL_OFFSET; ++i, ++ieta) {
    if (!m_EttMask[ieta]) {
      et0 += m_inputRegions.at(i).et();
      of  |= m_inputRegions.at(i).overFlow();
      et1 += m_inputRegions.at(i+COL_OFFSET).et();
      of  |= m_inputRegions.at(i+COL_OFFSET).overFlow();
    }
  }

  etTotalType etStrip0(et0);
  etTotalType etStrip1(et1);
  etStrip0.setOverFlow(etStrip0.overFlow() || of);
  etStrip1.setOverFlow(etStrip1.overFlow() || of);
  unsigned xfact0 = (4*m_id +  6) % 36;
  unsigned xfact1 = (4*m_id +  8) % 36;
  unsigned yfact0 = (4*m_id + 15) % 36;
  unsigned yfact1 = (4*m_id + 17) % 36;
  m_outputEtSum = etStrip0 + etStrip1;
  if (m_outputEtSum.overFlow()) m_outputEtSum.setValue(etTotalMaxValue);
  m_outputExSum = etComponentForJetFinder<L1GctInternEtSum::kTotEtOrHtNBits,L1GctInternEtSum::kJetMissEtNBits>
    (etStrip0, xfact0, etStrip1, xfact1);
  m_outputEySum = etComponentForJetFinder<L1GctInternEtSum::kTotEtOrHtNBits,L1GctInternEtSum::kJetMissEtNBits>
    (etStrip0, yfact0, etStrip1, yfact1);

  m_outputEtSumPipe.store(m_outputEtSum, bxRel());
  m_outputExSumPipe.store(m_outputExSum, bxRel());
  m_outputEySumPipe.store(m_outputEySum, bxRel());
}

// Calculates scalar and vector sum of Ht over calibrated jets
void L1GctJetFinderBase::doHtSums() {
  unsigned htt = 0;
  unsigned ht0 = 0;
  unsigned ht1 = 0;
  bool of = false;

  for(UShort i=0; i < MAX_JETS_OUT; ++i)
  {
    // Only sum Ht for valid jets
    if (!m_outputJets.at(i).isNullJet()) {
      unsigned ieta  = m_outputJets.at(i).rctEta();
      unsigned htJet = m_outputJets.at(i).calibratedEt(m_jetEtCalLuts.at(ieta));
      // Scalar sum of Htt, with associated threshold
      if (htJet >= m_HttSumJetThreshold && !m_HttMask[ieta]) {
	htt += htJet;
      } 
      // Strip sums, for input to Htm calculation, with associated threshold
      if (htJet >= m_HtmSumJetThreshold && !m_HtmMask[ieta]) {
	if (m_outputJets.at(i).rctPhi() == 0) {
	  ht0 += htJet;
	}
	if (m_outputJets.at(i).rctPhi() == 1) {
	  ht1 += htJet;
	}
	of |= m_outputJets.at(i).overFlow();
      }
    }
  }

  etHadType httTotal(htt);
  etHadType htStrip0(ht0);
  etHadType htStrip1(ht1);
  httTotal.setOverFlow(httTotal.overFlow() || of);
  if (httTotal.overFlow()) httTotal.setValue(htTotalMaxValue);
  htStrip0.setOverFlow(htStrip0.overFlow() || of);
  htStrip1.setOverFlow(htStrip1.overFlow() || of);
  unsigned xfact0 = (4*m_id + 10) % 36;
  unsigned xfact1 = (4*m_id +  4) % 36;
  unsigned yfact0 = (4*m_id + 19) % 36;
  unsigned yfact1 = (4*m_id + 13) % 36;
  m_outputHtSum = httTotal;
  m_outputHxSum = etComponentForJetFinder<L1GctInternEtSum::kTotEtOrHtNBits,L1GctInternHtMiss::kJetMissHtNBits>
    (htStrip0, xfact0, htStrip1, xfact1);
  m_outputHySum = etComponentForJetFinder<L1GctInternEtSum::kTotEtOrHtNBits,L1GctInternHtMiss::kJetMissHtNBits>
    (htStrip0, yfact0, htStrip1, yfact1);

  // Common overflow for Ht components
  bool htmOverFlow = m_outputHxSum.overFlow() || m_outputHySum.overFlow();
  m_outputHxSum.setOverFlow(htmOverFlow);
  m_outputHySum.setOverFlow(htmOverFlow);

  m_outputHtSumPipe.store(m_outputHtSum, bxRel());
  m_outputHxSumPipe.store(m_outputHxSum, bxRel());
  m_outputHySumPipe.store(m_outputHySum, bxRel());
}


// Calculates Hf inner rings Et sum, and counts number of "fineGrain" bits set
L1GctJetFinderBase::hfTowerSumsType L1GctJetFinderBase::calcHfSums() const
{
  static const UShort NUMBER_OF_FRWRD_RINGS = 4;
  static const UShort NUMBER_OF_INNER_RINGS = 2;
  std::vector<unsigned> et(NUMBER_OF_INNER_RINGS, 0);
  std::vector<bool>     of(NUMBER_OF_INNER_RINGS, false);
  std::vector<unsigned> nt(NUMBER_OF_INNER_RINGS, 0);

  UShort offset = COL_OFFSET*(centralCol0() + 1);
  for (UShort i=0; i < NUMBER_OF_FRWRD_RINGS; ++i) {
    offset--;

    // Sum HF Et and count jets above threshold over "inner rings"
    if (i<NUMBER_OF_INNER_RINGS) {
      et.at(i) += m_inputRegions.at(offset).et();
      of.at(i) = of.at(i) || m_inputRegions.at(offset).overFlow();

      et.at(i) += m_inputRegions.at(offset+COL_OFFSET).et();
      of.at(i) = of.at(i) || m_inputRegions.at(offset+COL_OFFSET).overFlow();

      if (m_inputRegions.at(offset).fineGrain()) nt.at(i)++;
      if (m_inputRegions.at(offset+COL_OFFSET).fineGrain()) nt.at(i)++;
    }
  }
  hfTowerSumsType temp(et.at(0), et.at(1), nt.at(0), nt.at(1));
  temp.etSum0.setOverFlow(temp.etSum0.overFlow() || of.at(0));
  temp.etSum1.setOverFlow(temp.etSum1.overFlow() || of.at(1));
  return temp;
}


// Here is where the rotations are actually done
// Procedure suitable for implementation in hardware, using
// integer multiplication and bit shifting operations

template <int kBitsInput, int kBitsOutput>
L1GctTwosComplement<kBitsOutput>
L1GctJetFinderBase::etComponentForJetFinder(const L1GctUnsignedInt<kBitsInput>& etStrip0, const unsigned& fact0,
					    const L1GctUnsignedInt<kBitsInput>& etStrip1, const unsigned& fact1) {

  // typedefs and constants
  typedef L1GctTwosComplement<kBitsOutput> OutputType;

  // The sin(phi), cos(phi) factors are represented in 15 bits, 
  // as numbers in the range -2^14 to 2^14.
  // We multiply each input strip Et by the required factor
  // then shift, to divide by 2^13. This gives an extra bit
  // of precision on the LSB of the output values.
  // It's important to avoid systematically biasing the Ex, Ey
  // component values because this results in an asymmetric
  // distribution in phi for the final MEt.
  // The extra LSB is required because one of the factors is 0.5.
  // Applying this factor without the extra LSB corrects odd values
  // systematically down by 0.5; or all values by 0.25
  // on average, giving a shift of -2 units in Ex.

  static const int internalComponentSize = 15;
  static const int maxEt                 = 1<<internalComponentSize;

  static const int kBitsFactor           = internalComponentSize+kBitsInput+1;
  static const int maxFactor             = 1<<kBitsFactor;

  static const int bitsToShift           = internalComponentSize-2;
  static const int halfInputLsb          = 1<<(bitsToShift-1);

  // These factors correspond to the sine of angles from -90 degrees to
  // 90 degrees in 10 degree steps, multiplied by 16383 and written
  // as a <kBitsFactor>-bit 2s-complement number.
  const int factors[19] = {maxFactor-16383, maxFactor-16134, maxFactor-15395, maxFactor-14188, maxFactor-12550,
			   maxFactor-10531,  maxFactor-8192,  maxFactor-5603,  maxFactor-2845, 0,
			   2845, 5603, 8192, 10531, 12550, 14188, 15395, 16134, 16383};

  int rotatedValue0, rotatedValue1, myFact;
  int etComponentSum = 0;

  if (fact0 >= 36 || fact1 >= 36) {
    if (m_verbose) {
      edm::LogError("L1GctProcessingError")
	<< "L1GctJetLeafCard::rotateEtValue() has been called with factor numbers "
	<< fact0 << " and " << fact1 << "; should be less than 36 \n";
    } 
  } else {

    // First strip - choose the required multiplication factor
    if (fact0>18) { myFact = factors[(36-fact0)]; }
    else { myFact = factors[fact0]; }

    // Multiply the Et value by the factor.
    rotatedValue0 = static_cast<int>(etStrip0.value()) * myFact;

    // Second strip - choose the required multiplication factor
    if (fact1>18) { myFact = factors[(36-fact1)]; }
    else { myFact = factors[fact1]; }

    // Multiply the Et value by the factor.
    rotatedValue1 = static_cast<int>(etStrip1.value()) * myFact;

    // Add the two scaled values together, with full resolution including
    // fractional parts from the sin(phi), cos(phi) scaling.
    // Adjust the value to avoid truncation errors since these
    // accumulate and cause problems for the missing Et measurement.
    // Then discard the 13 LSB and interpret the result as
    // a 15-bit twos complement integer.
    etComponentSum = ((rotatedValue0 + rotatedValue1) + halfInputLsb)>>bitsToShift;

    etComponentSum = etComponentSum & (maxEt-1);
    if (etComponentSum >= (maxEt/2)) {
      etComponentSum = etComponentSum - maxEt;
    }
  }

  // Store as a TwosComplement format integer and return
  OutputType temp(etComponentSum);
  temp.setOverFlow(temp.overFlow() || etStrip0.overFlow() || etStrip1.overFlow());
  return temp;
}

// Declare the specific versions we want to use, to help the linker out
// One for the MET components
template
L1GctJetFinderBase::etCompInternJfType
L1GctJetFinderBase::etComponentForJetFinder<L1GctInternEtSum::kTotEtOrHtNBits,L1GctInternEtSum::kJetMissEtNBits>
(const L1GctJetFinderBase::etTotalType&, const unsigned&,
 const L1GctJetFinderBase::etTotalType&, const unsigned&);

// One for the MHT components
template
L1GctJetFinderBase::htCompInternJfType
L1GctJetFinderBase::etComponentForJetFinder<L1GctInternEtSum::kTotEtOrHtNBits,L1GctInternHtMiss::kJetMissHtNBits>
(const L1GctJetFinderBase::etTotalType&, const unsigned&,
 const L1GctJetFinderBase::etTotalType&, const unsigned&);
