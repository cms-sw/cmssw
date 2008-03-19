#include "EventFilter/GctRawToDigi/src/GctBlockUnpackerV2.h"

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
GctBlockUnpackerV2::BlockIdToUnpackFnMap GctBlockUnpackerV2::blockUnpackFn_ = GctBlockUnpackerV2::BlockIdToUnpackFnMap();

GctBlockUnpackerV2::GctBlockUnpackerV2(bool hltMode):
  GctBlockUnpackerBase(hltMode),
{
  static bool initClass = true;

  if(initClass)
  {
    initClass = false;

    // Setup block unpack function map
    blockUnpackFn_[0x000] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x583] = &GctBlockUnpackerV2::blockToGctJetCandsAndCounts;
    blockUnpackFn_[0x580] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x587] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x683] = &GctBlockUnpackerV2::blockToGctEmCandsAndEnergySums;
    blockUnpackFn_[0x680] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0x686] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x687] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x800] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0x804] = &GctBlockUnpackerBase::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0x803] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0x880] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0x884] = &GctBlockUnpackerBase::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0x883] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0xc00] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0xc04] = &GctBlockUnpackerBase::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0xc03] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0xc80] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0xc84] = &GctBlockUnpackerBase::blockToFibresAndToRctEmCand;
    blockUnpackFn_[0xc83] = &GctBlockUnpackerBase::blockToGctInternEmCand;
    blockUnpackFn_[0x900] = &GctBlockUnpackerBase::blockDoNothing;  // Start of leafJet blocks (unknown how to unpack at the moment)
    blockUnpackFn_[0x904] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x901] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x902] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x903] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x908] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x90c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x909] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x90a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x90b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x980] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x984] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x988] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x98c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x989] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x98a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0x98b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa00] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa04] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa01] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa02] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa03] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa08] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa0c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa09] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa0a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa0b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa80] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa84] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa88] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa8c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa89] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa8a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xa8b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb00] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb04] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb01] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb02] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb03] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb08] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb0c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb09] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb0a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb0b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb80] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb84] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb88] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb8c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb89] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb8a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xb8b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd00] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd04] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd01] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd02] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd03] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd08] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd0c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd09] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd0a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd0b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd80] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd84] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd88] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd8c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd89] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd8a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xd8b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe00] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe04] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe01] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe02] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe03] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe08] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe0c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe09] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe0a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe0b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe80] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe84] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe88] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe8c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe89] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe8a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xe8b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf00] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf04] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf01] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf02] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf03] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf08] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf0c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf09] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf0a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf0b] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf80] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf84] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf88] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf8c] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf89] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf8a] = &GctBlockUnpackerBase::blockDoNothing;
    blockUnpackFn_[0xf8b] = &GctBlockUnpackerBase::blockDoNothing;  // End of leafJet blocks.
    blockUnpackFn_[0x0ff] = &GctBlockUnpackerBase::blockToRctCaloRegions;  // Our temp hack RCT calo block
  }
}

GctBlockUnpackerV2::~GctBlockUnpackerV2() { }

// conversion
void GctBlockUnpackerV2::convertBlock(const unsigned char * data, const GctBlockHeaderBase& hdr)
{
  if(!checkBlock(hdr)) { return; }  // Check the block to see if it's possible to unpack.

  // The header validity check above will protect against
  // the map::find() method returning the end of the map,
  // assuming the block header definitions are up-to-date.
  (this->*blockUnpackFn_.find(id)->second)(data, hdr);  // Calls the correct unpack function, based on block ID.
}

// Output EM Candidates unpacking
void GctBlockUnpackerV2::blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  LogDebug("GCT") << "Unpacking GCT output EM Cands and Energy Sums" << std::endl;

  const unsigned int id = hdr.id();
  const unsigned int nSamples = hdr.nSamples();

  // Re-interpret pointer.  p16 will be pointing at the 16 bit word that
  // contains the rank0 non-isolated electron of the zeroth time-sample.
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);

  // UNPACK EM CANDS

  const unsigned int emCandCatagoryOffset = nSamples * 4;  // Offset to jump from the non-iso electrons to the isolated ones.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  for (unsigned int iso=0; iso<2; ++iso)  // loop over non-iso/iso candidate pairs
  {
    bool isoFlag = (iso==1);

    // Get the correct collection to put them in.
    L1GctEmCandCollection* em;
    if (isoFlag) { em = gctIsoEm_; }
    else { em = gctNonIsoEm_; }

    for (unsigned int bx=0; bx<nSamples; ++bx) // loop over time samples
    {
      // cand0Offset will give the offset on p16 to get the rank 0 candidate
      // of the correct catagory and timesample.
      const unsigned int cand0Offset = iso*emCandCatagoryOffset + bx*2;

      em->push_back(L1GctEmCand(p16[cand0Offset], isoFlag, id, 0, bx));  // rank0 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + timeSampleOffset], isoFlag, id, 1, bx));  // rank1 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + 1], isoFlag, id, 2, bx));  // rank2 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + timeSampleOffset + 1], isoFlag, id, 3, bx));  // rank3 electron
    }
  }

  p16 += emCandCatagoryOffset * 2;  // Move the pointer over the data we've already unpacked.

  // UNPACK ENERGY SUMS
  /* NOTE: we are only unpacking one timesample of these, because the
   * relevant dataformats do not have timesample support yet. */

  *gctEtTotal_ = L1GctEtTotal(p16[0]);  // Et total (timesample 0).
  *gctEtHad_ = L1GctEtHad(p16[1]);  // Et hadronic (timesample 0).

  // 32-bit pointer for getting Missing Et.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  *gctEtMiss_ = L1GctEtMiss(p32[nSamples]); // Et Miss (timesample 0).
}

void GctBlockUnpackerV2::blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  LogDebug("GCT") << "Unpacking GCT output Jet Cands and Counts" << std::endl;

  const unsigned int id = hdr.id();  // Capture block ID.
  const unsigned int nSamples = hdr.nSamples();  // Number of time-samples.

  // Re-interpret block payload pointer to 16 bits so it sees one candidate at a time.
  // p16 points to the start of the block payload, at the rank0 tau jet candidate.
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);

  // UNPACK JET CANDS

  const unsigned int jetCandCatagoryOffset = nSamples * 4;  // Offset to jump from one jet catagory to the next.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  // Loop over the different catagories of jets
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATAGORIES ; ++iCat)
  {
    assert(gctJets_.at(iCat)->empty()); // The supplied vector should be empty.

    bool tauflag = (iCat == TAU_JETS);
    bool forwardFlag = (iCat == FORWARD_JETS);

    // Loop over the different timesamples (bunch crossings).
    for(unsigned int bx = 0 ; bx < nSamples ; ++bx)
    {
      // cand0Offset will give the offset on p16 to get the rank 0 Jet Cand of the correct catagory and timesample.
      const unsigned int cand0Offset = iCat*jetCandCatagoryOffset + bx*2;

      // Rank 0 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset], tauflag, forwardFlag, id, 0, bx));
      // Rank 1 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset], tauflag, forwardFlag, id, 1, bx));
      // Rank 2 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset + 1],  tauflag, forwardFlag, id, 2, bx));
      // Rank 3 Jet.
      gctJets_.at(iCat)->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset + 1], tauflag, forwardFlag, id, 3, bx));
    }
  }

  p16 += NUM_JET_CATAGORIES * jetCandCatagoryOffset; // Move the pointer over the data we've already unpacked.

  // UNPACK JET COUNTS
  /* NOTE: we are only unpacking one timesample of these, because the
   * dataformat for jet counts does not have timesample support yet. */

  // Re-interpret block payload pointer to 32 bits so it sees six jet counts at a time.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  // nSamples below gives the offset to the second set of six jet counts in timesample 0.
  *gctJetCounts_ = L1GctJetCounts(p32[0], p32[nSamples]);
}
