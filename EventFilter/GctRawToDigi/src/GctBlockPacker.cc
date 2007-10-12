
#include "EventFilter/GctRawToDigi/src/GctBlockPacker.h"
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"


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


// Output EM Candidates packing
void GctBlockPacker::writeGctEmBlock(unsigned char * d, const L1GctEmCandCollection* iso,
                                     const L1GctEmCandCollection* nonIso)
{
  // number of time samples to write
  uint16_t nSamples = 1;

  // write header
  writeGctHeader(d, 0x68, nSamples);
  d=d+4;  // move to the block payload.

  // re-interpret pointer
  uint16_t * p = reinterpret_cast<uint16_t *>(d);

  for (int i=0; i<2; i++) {   // loop over non-iso/iso candidates
    for (int bx=0; bx<nSamples; bx++) {   // loop over time samples

      bool isolated = (i==1);
      uint16_t * pp = p+ (2*bx) + (i*4*nSamples);
      const L1GctEmCandCollection* em;
      
      if (isolated) { em = iso; }
      else { em = nonIso; }

      *pp = em->at(0).raw();
      pp++;
      *pp = em->at(2).raw();
      pp=pp+nSamples;
      *pp = em->at(1).raw();
      pp++;
      *pp = em->at(3).raw();
      
    }
  }

}

void GctBlockPacker::writeGctJetBlock(unsigned char * d,
                                      const L1GctEmCandCollection* cenJets,
                                      const L1GctEmCandCollection* forJets,
                                      const L1GctEmCandCollection* tauJets)
{
  // Set up a vector of the collections for easy iteration.
  vector<L1GctEmCandCollection*> jets(NUM_JET_CATAGORIES);
  vector.at(CENTRAL_JETS)=cenJets;
  vector.at(FORWARD_JETS)=forJets;
  vector.at(TAU_JETS)=tauJets;
  
  // number of time samples to write
  const uint16_t nSamples = 1;
  const unsigned int catagoryOffset = nSamples * 4;  // Offset to jump from one jet catagory to the next.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  // write header
  writeGctHeader(d, 0x58, nSamples);
  d=d+4;  // move forward past the block header to the block payload.

  // re-interpret pointer to 16 bits - the space allocated for each Jet candidate.
  uint16_t * p = reinterpret_cast<uint16_t *>(d);
  
  // Loop over the different catagories of jets
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATAGORIES ; ++iCat)
  {
    // Loop over the different timesamples (bunch crossings).    
    for(unsigned int bx = 0 ; bx < nSamples ; ++bx)
    {
      // cand0Offset will give the offset on p to get the rank 0 Jet Cand of the correct catagory and timesample.
      const unsigned int cand0Offset = iCat*catagoryOffset + bx*2;
      
      p[cand0Offset] = jets.at(iCat)->at(0).raw();  // rank 0 jet
      p[cand0Offset + timeSampleOffset] = jets.at(iCat)->(1).raw(); // rank 1 jet
      p[cand0Offset + 1] = jets.at(iCat)->at(2).raw(); // rank 2 jet
      p[cand0Offset + timeSampleOffset + 1] = jet.at(iCat)->at(3).raw(); // rank 3 jet.
    }
  }
}



// NOTE : all methods below are placeholders for use before the real raw format is defined!
// In particular, the block IDs are complete fiction

// Output Cen Jet Candidates packing
void GctBlockPacker::writeGctCenJetBlock(unsigned char * d, const L1GctJetCandCollection* coll) {

  // write header
  writeGctHeader(d, 0x01, 1);
  d=d+4;

  // cast to 16 bit pointer
  uint16_t * p = reinterpret_cast<uint16_t*>(const_cast<unsigned char *>(d));

   // pack jets
    for (int i=0; i<4; i++) {
//      // in future, will only pack digis for 0th crossing, but this is not set yet!!!
//      //    if (gctIsoEm_->at(i).bx() == 0) {
        *p = coll->at(i).raw();
        p++;
        //    }
    }
  
}


// Output EM Candidates packing
void GctBlockPacker::writeGctTauJetBlock(unsigned char * d, const L1GctJetCandCollection* coll) {

  // write header
  writeGctHeader(d, 0x02, 1);
  d=d+4;

  // cast to 16 bit pointer
  uint16_t * p = reinterpret_cast<uint16_t*>(const_cast<unsigned char *>(d));

   // pack jets
    for (int i=0; i<4; i++) {
//      // in future, will only pack digis for 0th crossing, but this is not set yet!!!
//      //    if (gctIsoEm_->at(i).bx() == 0) {
        *p = coll->at(i).raw();
        p++;
        //    }
    }
  
}


// Output EM Candidates packing
void GctBlockPacker::writeGctForJetBlock(unsigned char * d, const L1GctJetCandCollection* coll) {

  // write header
  writeGctHeader(d, 0x03, 1);
  d=d+4;

  // cast to 16 bit pointer
  uint16_t * p = reinterpret_cast<uint16_t*>(const_cast<unsigned char *>(d));

   // pack jets
    for (int i=0; i<4; i++) {
//      // in future, will only pack digis for 0th crossing, but this is not set yet!!!
//      //    if (gctIsoEm_->at(i).bx() == 0) {
        *p = coll->at(i).raw();
        p++;
        //    }
    }

}

// Output Energy Sums packin
void GctBlockPacker::writeEnergySumsBlock(unsigned char * d, const L1GctEtMiss* etm, const L1GctEtTotal* ett, const L1GctEtHad* eth) {

  // write header
  writeGctHeader(d, 0x04, 1);
  d=d+4;

}

