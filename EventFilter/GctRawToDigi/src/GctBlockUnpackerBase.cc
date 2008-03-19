#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerBase.h"

// C++ headers
#include <iostream>
#include <cassert>

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Namespace resolution
using std::cout;
using std::endl;
using std::pair;

// INITIALISE STATIC VARIABLES
GctBlockUnpackerBase::RctCrateMap GctBlockUnpackerBase::rctCrate_ = GctBlockUnpackerBase::RctCrateMap();
GctBlockUnpackerBase::BlockIdToEmCandIsoBoundMap GctBlockUnpackerBase::InternEmIsoBounds_ = GctBlockUnpackerBase::BlockIdToEmCandIsoBoundMap();

GctBlockUnpackerBase::GctBlockUnpackerBase(bool hltMode):
  hltMode_(hltMode),
  srcCardRouting_(),
  rctEm_(0),
  rctCalo_(0),
  gctIsoEm_(0),
  gctNonIsoEm_(0),
  gctInternEm_(0),
  gctFibres_(0),
  gctJets_(NUM_JET_CATAGORIES),
  gctJetCounts_(0),
  gctEtTotal_(0),
  gctEtHad_(0),
  gctEtMiss_(0)
{
  static bool initClass = true;
  
  if(initClass)
  {
    initClass = false;
    
    rctCrate_[0x81] = 13;
    rctCrate_[0x89] = 9;
    rctCrate_[0xC1] = 4;
    rctCrate_[0xC9] = 0; 

    // Setup Block ID map for pipeline payload positions of isolated Internal EM Cands.
    InternEmIsoBounds_[0x69] = IsoBoundaryPair(8,15);
    InternEmIsoBounds_[0x80] = IsoBoundaryPair(0, 9);
    InternEmIsoBounds_[0x83] = IsoBoundaryPair(0, 1);
    InternEmIsoBounds_[0x88] = IsoBoundaryPair(0, 7);
    InternEmIsoBounds_[0x8b] = IsoBoundaryPair(0, 1);
    InternEmIsoBounds_[0xc0] = IsoBoundaryPair(0, 9);
    InternEmIsoBounds_[0xc3] = IsoBoundaryPair(0, 1);
    InternEmIsoBounds_[0xc8] = IsoBoundaryPair(0, 7);
    InternEmIsoBounds_[0xcb] = IsoBoundaryPair(0, 1);
  }
}

GctBlockUnpackerBase::~GctBlockUnpackerBase() { }

// conversion
bool GctBlockUnpackerBase::checkBlock(const GctBlockHeaderBase& hdr)
{
  unsigned int nSamples = hdr.nSamples();

  // if the block has no time samples, don't bother
  if ( nSamples < 1 ) { return false; }

  // check block is valid
  if ( !hdr.valid() )
  {
    edm::LogError("GCT") << "Attempting to unpack an unidentified block\n" << hdr << endl;
    return false;     
  }

  // check block doesn't have too many time samples
  if ( nSamples >= 0xf ) {
    edm::LogError("GCT") << "Cannot unpack a block with 15 or more time samples\n" << hdr << endl;
    return false; 
  }
  return true;
}

// Internal EM Candidates unpacking
void GctBlockUnpackerBase::blockToGctInternEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal EM Cands" << std::endl; return; }
  
  LogDebug("GCT") << "Unpacking internal EM Cands" << std::endl;

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int numCandPairs = hdr.length();

  // Debug assertion to prevent problems if definitions not up to date.
  assert(InternEmIsoBounds_.find(id) != InternEmIsoBounds_.end());  

  unsigned int lowerIsoPairBound = InternEmIsoBounds_[id].first;
  unsigned int upperIsoPairBound = InternEmIsoBounds_[id].second;

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

  // Loop over timesamples (i.e. bunch crossings)
  for(unsigned int bx=0; bx < nSamples; ++bx)
  {
    // Loop over candidate pairs (i.e. each iteration unpacks a pair of candidates)
    for(unsigned int candPair = 0 ; candPair < numCandPairs ; ++candPair)
    {
      // Is the candidate electron pair an isolated pair or not?
      bool iso = ((candPair>=lowerIsoPairBound) && (candPair<=upperIsoPairBound));
      
      // Loop over the two electron candidates in each pair
      for(unsigned int i = 0 ; i < 2 ; ++i)
      { 
        unsigned offset = 2*(bx + candPair*nSamples) + i;
        uint16_t candRaw = p[offset]; 
        gctInternEm_->push_back( L1GctInternEmCand(candRaw, iso, id, candPair*2 + i, bx) );
      }
    }
  }
}


// Input EM Candidates unpacking
// this is the last time I deal the RCT bit assignment travesty!!!
void GctBlockUnpackerBase::blockToRctEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT EM Cands" << std::endl; return; }
  
  LogDebug("GCT") << "Unpacking RCT EM Cands" << std::endl;

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // re-interpret pointer
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

  // arrays of source card data
  uint16_t sfp[2][4]; // [ cycle ] [ SFP ]
  uint16_t eIsoRank[4];
  uint16_t eIsoCard[4];
  uint16_t eIsoRgn[4];
  uint16_t eNonIsoRank[4];
  uint16_t eNonIsoCard[4];
  uint16_t eNonIsoRgn[4];
  uint16_t MIPbits[7][2];
  uint16_t QBits[7][2];

  unsigned int bx = 0;

  // loop over crates
  for (unsigned int crate=rctCrate_[id]; crate<rctCrate_[id]+length/3; ++crate) {

    // read SC SFP words
    for (unsigned short iSfp=0 ; iSfp<4 ; ++iSfp) {
      for (unsigned short cyc=0 ; cyc<2 ; ++cyc) {
        if (iSfp==0) { sfp[cyc][iSfp] = 0; } // muon bits
        else {                               // EM candidate
          sfp[cyc][iSfp] = *p;
          ++p;
        }
      }
      p = p + 2*(nSamples-1);
    }

    // fill SC arrays
    srcCardRouting_.SFPtoEMU(eIsoRank, eIsoCard, eIsoRgn, eNonIsoRank, eNonIsoCard, eNonIsoRgn, MIPbits, QBits, sfp);
    
    // create EM cands
    for (unsigned short int i=0; i<4; ++i) {
      rctEm_->push_back( L1CaloEmCand( eIsoRank[i], eIsoRgn[i], eIsoCard[i], crate, true, i, bx) );
    }
    for (unsigned short int i=0; i<4; ++i) {
      rctEm_->push_back( L1CaloEmCand( eNonIsoRank[i], eNonIsoRgn[i], eNonIsoCard[i], crate, false, i, bx) );
    }
  }
}


// Fibre unpacking
void GctBlockUnpackerBase::blockToFibres(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of GCT Fibres" << std::endl; return; }

  LogDebug("GCT") << "Unpacking GCT Fibres" << std::endl;
  
  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // re-interpret pointer
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      gctFibres_->push_back( L1GctFibreWord(*p, id, i, bx) );
      ++p;
    }
  } 
}

void GctBlockUnpackerBase::blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  this->blockToRctEmCand(d, hdr);
  this->blockToFibres(d, hdr);
}

void GctBlockUnpackerBase::blockToRctCaloRegions(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT Calo Regions" << std::endl; return; }

  // This method is one giant "temporary" hack whilst waiting for proper
  // pipeline formats for the RCT calo region data.
  
  LogDebug("GCT") << "Unpacking RCT Calorimeter Regions" << std::endl;
  
  const int nSamples = hdr.nSamples();  // Number of time-samples.

  // Re-interpret block payload pointer to 16 bits
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);
  
  for(unsigned iCrate = 0 ; iCrate < 18 ; ++iCrate)
  {
    // Barrel and endcap regions
    for(unsigned iCard = 0 ; iCard < 7 ; ++iCard)
    {
      // Samples
      for(int16_t iSample = 0 ; iSample < nSamples ; ++iSample)
      {
        // Two regions per card (and per 32-bit word).
        for(unsigned iRegion = 0 ; iRegion < 2 ; ++iRegion)
        {
          L1CaloRegionDetId id(iCrate, iCard, iRegion);
          rctCalo_->push_back(L1CaloRegion(*p16, id.ieta(), id.iphi(), iSample));
          ++p16; //advance pointer
        }
      }
    }
    // Forward regions (8 regions numbered 0 through 7, packed in 4 sets of pairs)
    for(unsigned iRegionPairNum = 0 ; iRegionPairNum < 4 ; ++iRegionPairNum)
    {
      // Samples
      for(int16_t iSample = 0 ; iSample < nSamples ; ++iSample)
      {
        // two regions in a pair
        for(unsigned iPair = 0 ; iPair < 2 ; ++iPair)
        {
          // For forward regions, RCTCard=999
          L1CaloRegionDetId id(iCrate, 999, iRegionPairNum*2 + iPair);
          rctCalo_->push_back(L1CaloRegion(*p16, id.ieta(), id.iphi(), iSample));
          ++p16; //advance pointer
        }
      }
    }
  }
}
