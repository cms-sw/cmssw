
#include "EventFilter/GctRawToDigi/src/GctBlockPacker.h"
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"


GctBlockPacker::GctBlockPacker() {

}

GctBlockPacker::~GctBlockPacker() {

}

// write CDF FED header
void GctBlockPacker::writeFedHeader(unsigned char * d, uint32_t fedId) {

  uint64_t hdr = 0x18
    + ((uint64_t)(fedId & 0xFFF)<<8)
    + ((uint64_t)((uint64_t)bcid_ & 0xFFF)<<20)
    + ((uint64_t)((uint64_t)evid_ & 0xFFFFFF)<<32)
    + ((uint64_t)((uint64_t)0x51<<56));

  uint64_t * p = reinterpret_cast<uint64_t *>(const_cast<unsigned char *>(d));
  *p = hdr;
  
}


// write FED Footer
void GctBlockPacker::writeFedFooter(unsigned char * d, const unsigned char * s) {

}


// write GCT internal header
void GctBlockPacker::writeGctHeader(unsigned char * d, uint16_t id, uint16_t nsamples) {
  
  uint32_t hdr = GctBlockHeader(id, nsamples, bcid_, evid_).data();
  uint32_t * p = reinterpret_cast<uint32_t*>(const_cast<unsigned char *>(d));
  *p = hdr;

}


// Output EM Candidates packing
void GctBlockPacker::writeGctEmBlock(unsigned char * d, const L1GctEmCandCollection* iso, const L1GctEmCandCollection* nonIso) {

  // number of time samples to write
  uint16_t nSamples = 1;

  // write header
  writeGctHeader(d, 0x68, nSamples);
  d=d+4;

  // re-interpret pointer
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

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

