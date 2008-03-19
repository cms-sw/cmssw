#include "EventFilter/GctRawToDigi/src/GctBlockUnpacker.h"

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
GctBlockUnpacker::BlockIdToUnpackFnMap GctBlockUnpacker::blockUnpackFn_ = GctBlockUnpacker::BlockIdToUnpackFnMap();


// PUBLIC METHODS

GctBlockUnpacker::GctBlockUnpacker(bool hltMode):
  GctBlockUnpackerBase(hltMode)
{
  static bool initClass = true;
  
  if(initClass)
  {
    initClass = false;

    // Setup block unpack function map
    blockUnpackFn_[0x00] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x58] = &GctBlockUnpacker::blockToGctJetCand;
    blockUnpackFn_[0x59] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x5a] = &GctBlockUnpacker::blockToGctJetCounts;
    blockUnpackFn_[0x5f] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x68] = &GctBlockUnpacker::blockToGctEmCand;
    blockUnpackFn_[0x69] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0x6a] = &GctBlockUnpacker::blockToGctEnergySums;
    blockUnpackFn_[0x6b] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x6f] = &GctBlockUnpacker::blockDoNothing; 
    blockUnpackFn_[0x80] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0x81] = &GctBlockUnpacker::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0x83] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0x88] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0x89] = &GctBlockUnpacker::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0x8b] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0xc0] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0xc1] = &GctBlockUnpacker::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0xc3] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0xc8] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0xc9] = &GctBlockUnpacker::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0xcb] = &GctBlockUnpacker::blockToGctInternEmCand;
    blockUnpackFn_[0xff] = &GctBlockUnpacker::blockToRctCaloRegions;  // Our temp hack RCT calo block
  }
}

GctBlockUnpacker::~GctBlockUnpacker() { }

// conversion
void GctBlockUnpacker::convertBlock(const unsigned char * data, const GctBlockHeaderBase& hdr)
{
  if(!checkBlock(hdr)) { return; }  // Check the block to see if it's possible to unpack.

  // The header validity check above will protect against 
  // the map::find() method returning the end of the map,
  // assuming the block header definitions are up-to-date.
  (this->*blockUnpackFn_.find(hdr.id())->second)(data, hdr);  // Calls the correct unpack function, based on block ID.
}


// PRIVATE METHODS

void GctBlockUnpacker::blockToGctJetCand(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  LogDebug("GCT") << "Unpacking GCT output Jet Cands" << std::endl;
  
  const unsigned int id = hdr.id();  // Capture block ID.
  const unsigned int nSamples = hdr.nSamples();  // Number of time-samples.

  const unsigned int catagoryOffset = nSamples * 4;  // Offset to jump from one jet catagory to the next.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  unsigned int samplesToUnpack = 1;
  if(!hltMode()) { samplesToUnpack = nSamples; }  // Only if not running in HLT mode do we want more than 1 timesample. 

  // Re-interpret block payload pointer to 16 bits so it sees one candidate at a time.
  // p points to the start of the block payload, at the rank0 tau jet candidate.
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);
  
  // Loop over the different catagories of jets
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATAGORIES ; ++iCat)
  {
    assert(gctJets_.at(iCat)->empty()); // The supplied vector should be empty.

    bool tauflag = (iCat == TAU_JETS);
    bool forwardFlag = (iCat == FORWARD_JETS);
    
    // Loop over the different timesamples (bunch crossings).
    for(unsigned int bx = 0 ; bx < samplesToUnpack ; ++bx)
    {
      // cand0Offset will give the offset on p to get the rank 0 Jet Cand of the correct catagory and timesample.
      const unsigned int cand0Offset = iCat*catagoryOffset + bx*2;
      
      // Rank 0 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p[cand0Offset], tauflag, forwardFlag, id, 0, bx));
      // Rank 1 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p[cand0Offset + timeSampleOffset], tauflag, forwardFlag, id, 1, bx));
      // Rank 2 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p[cand0Offset + 1],  tauflag, forwardFlag, id, 2, bx));
      // Rank 3 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p[cand0Offset + timeSampleOffset + 1], tauflag, forwardFlag, id, 3, bx));      
    }
  }
}

void GctBlockUnpacker::blockToGctJetCounts(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  LogDebug("GCT") << "Unpacking GCT output Jet Counts" << std::endl;
  
  /*
   * Currently only unpacking one timesample of these!
   */
  
  // Re-interpret block payload pointer to 32 bits so it sees six jet counts at a time.
  // p points to the start of the block payload, at the first six jet counts in timesample 0.
  const uint32_t * p = reinterpret_cast<const uint32_t *>(d);

  // The call to hdr.nSamples() in the below line gives the offset from the start of the block
  // payload for a 32-bit pointer to get to the second set of six jet counts in timesample 0.
  *gctJetCounts_ = L1GctJetCounts(p[0], p[hdr.nSamples()]);
}

// Output EM Candidates unpacking
void GctBlockUnpacker::blockToGctEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  LogDebug("GCT") << "Unpacking GCT output EM Cands" << std::endl;

  const unsigned int id = hdr.id();
  const unsigned int nSamples = hdr.nSamples();

  const unsigned int catagoryOffset = nSamples * 4;  // Offset to jump from the non-iso electrons to the isolated ones.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  unsigned int samplesToUnpack = 1;
  if(!hltMode()) { samplesToUnpack = nSamples; }  // Only if not running in HLT mode do we want more than 1 timesample. 

  // Re-interpret pointer.  p will be pointing at the 16 bit word that
  // contains the rank0 non-isolated electron of the zeroth time-sample. 
  const uint16_t * p = reinterpret_cast<const uint16_t *>(d);

  for (unsigned int iso=0; iso<2; ++iso)  // loop over non-iso/iso candidate pairs
  {
    bool isoFlag = (iso==1);

    // Get the correct collection to put them in.
    L1GctEmCandCollection* em;
    if (isoFlag) { em = gctIsoEm_; }
    else { em = gctNonIsoEm_; }

    for (unsigned int bx=0 ; bx<samplesToUnpack ; ++bx) // loop over time samples
    {
      // cand0Offset will give the offset on p to get the rank 0 candidate
      // of the correct catagory and timesample.
      const unsigned int cand0Offset = iso*catagoryOffset + bx*2;
      
      em->push_back(L1GctEmCand(p[cand0Offset], isoFlag, id, 0, bx));  // rank0 electron
      em->push_back(L1GctEmCand(p[cand0Offset + timeSampleOffset], isoFlag, id, 1, bx));  // rank1 electron
      em->push_back(L1GctEmCand(p[cand0Offset + 1], isoFlag, id, 2, bx));  // rank2 electron
      em->push_back(L1GctEmCand(p[cand0Offset + timeSampleOffset + 1], isoFlag, id, 3, bx));  // rank3 electron
    }
  }
}

void GctBlockUnpacker::blockToGctEnergySums(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  LogDebug("GCT") << "Unpacking GCT output Energy Sums" << std::endl;
  
  /* 
   * Currently only unpacking one timesample of these!
   */
  
  // Re-interpret block payload pointer to both 16 and 32 bits.
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);  // For getting Et + Ht (16-bit raw data each)
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(d);  // For getting Missing Et (32-bit raw data)
  
  *gctEtTotal_ = L1GctEtTotal(p16[0]);  // Et total (first 16 bits in block payload for timesample 0)
  *gctEtHad_ = L1GctEtHad(p16[1]);  // Et hadronic (second 16 bits in block payload for timesample 0)
  
  // The call to hdr.nSamples() in the below line gives the offset from the start of the block
  // payload for a 32-bit pointer to get to the missing Et data in timesample 0.
  *gctEtMiss_ = L1GctEtMiss(p32[hdr.nSamples()]);    
}

