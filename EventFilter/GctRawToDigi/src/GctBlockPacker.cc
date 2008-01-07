#include "EventFilter/GctRawToDigi/src/GctBlockPacker.h"

// C++ headers
#include <vector>

// CMSSW headers
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

// Namespace resolution
using std::vector;


GctBlockPacker::GctBlockPacker() {

}


GctBlockPacker::~GctBlockPacker() {

}


// write GCT internal header
void GctBlockPacker::writeGctHeader(unsigned char * d, uint16_t id, uint16_t nsamples)
{
  uint32_t hdr = GctBlockHeader(id, nsamples, bcid_, evid_).data();
  uint32_t * p = reinterpret_cast<uint32_t*>(const_cast<unsigned char *>(d));
  *p = hdr;
}


void GctBlockPacker::writeGctOutJetBlock(unsigned char * d,
                                         const L1GctJetCandCollection* cenJets,
                                         const L1GctJetCandCollection* forJets,
                                         const L1GctJetCandCollection* tauJets, 
                                         const L1GctJetCounts* jetCounts)
{
  // write header
  writeGctHeader(d, 0x58, 1);
  
  d=d+4;  // move forward past the block header to the block payload.

  // FIRST DO JET CANDS
  // re-interpret pointer to 16 bits - the space allocated for each Jet candidate.
  uint16_t * p16 = reinterpret_cast<uint16_t *>(d);
  
  // Set up a vector of the collections for easy iteration.
  vector<const L1GctJetCandCollection*> jets(NUM_JET_CATAGORIES);
  jets.at(CENTRAL_JETS)=cenJets;
  jets.at(FORWARD_JETS)=forJets;
  jets.at(TAU_JETS)=tauJets;
  
  const unsigned int catagoryOffset = 4;  // Offset to jump from one jet catagory to the next.
  const unsigned int nextCandPairOffset = 2;  // Offset to jump to next candidate pair.

  // Loop over the different catagories of jets
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATAGORIES ; ++iCat)
  {
    // cand0Offset will give the offset on p16 to get the rank 0 Jet Cand of the correct catagory.
    const unsigned int cand0Offset = iCat*catagoryOffset;
    
    p16[cand0Offset] = jets.at(iCat)->at(0).raw();  // rank 0 jet
    p16[cand0Offset + nextCandPairOffset] = jets.at(iCat)->at(1).raw(); // rank 1 jet
    p16[cand0Offset + 1] = jets.at(iCat)->at(2).raw(); // rank 2 jet
    p16[cand0Offset + nextCandPairOffset + 1] = jets.at(iCat)->at(3).raw(); // rank 3 jet.
  }
  
  // NOW DO JET COUNTS
  d=d+24;  // move forward past the jet cands to the jet counts section

  // re-interpret pointer to 32 bit.
  uint32_t * p32 = reinterpret_cast<uint32_t *>(d);
  
  p32[0] = jetCounts->raw0();
  p32[1] = jetCounts->raw1();  
}

// Output EM Candidates packing
void GctBlockPacker::writeGctOutEmAndEnergyBlock(unsigned char * d,
                                                 const L1GctEmCandCollection* iso,
                                                 const L1GctEmCandCollection* nonIso,
                                                 const L1GctEtTotal* etTotal,
                                                 const L1GctEtHad* etHad,
                                                 const L1GctEtMiss* etMiss)
{
  unsigned nSamples = 1; // Note: can only currently do SINGLE TIME SAMPLE!
  
  // write header
  writeGctHeader(d, 0x68, nSamples);  
  
  d=d+4;  // move to the block payload.

  // FIRST DO EM CANDS

  // re-interpret payload pointer to 16 bit.
  uint16_t * p16 = reinterpret_cast<uint16_t *>(d);

  for (int i=0; i<2; ++i)   // loop over non-iso/iso candidates
  {
    bool isolated = (i==1);
    uint16_t * cand = p16 + (i*4);
    const L1GctEmCandCollection* em;
    
    if (isolated) { em = iso; }
    else { em = nonIso; }

    *cand = em->at(0).raw();
    cand++;
    *cand = em->at(2).raw();
    cand+=nSamples;
    *cand = em->at(1).raw();
    cand++;
    *cand = em->at(3).raw();
  }
  
  // NOW DO ENERGY SUMS
  
  p16+=8;  // Move past EM cands
  *p16 = etTotal->raw();  // Et Total - 16 bits.
  p16++;
  *p16 = etHad->raw();  // Et Hadronic - next 16 bits
  p16++;
  uint32_t * p32 = reinterpret_cast<uint32_t *>(p16);  // For writing Missing Et (32-bit raw data)
  *p32 = etMiss->raw();  // Et Miss on final 32 bits of block payload.
}

