
#include "EventFilter/GctRawToDigi/src/GctBlockPacker.h"

// Output EM Candidates packing
void GctBlockPacker::writeGctEmBlock(unsigned char * d, unsigned id) {

  // write header
  unsigned last = 0;
  writeGctHeader(d, 0x68);
  last += 4;

  // pack iso EM
  for (int i=0; i<4; i++) {
    // in future, will only pack digis for 0th crossing, but this is not set yet!!!
    //    if (gctIsoEm_->at(i).bx() == 0) {
      int j = i; // should be gctIsoEm_->at(i).capIndex(); but capIndex is not set yet!!!
      d[last] = gctIsoEm_->at(i).raw() & 0xff;
      last++;
      d[last] = (gctIsoEm_->at(i).raw()>>8) & 0xff;
      last++;
      //    }
  }

  // pack non-iso EM
  for (int i=0; i<4; i++) {
    // in future will ony pack digis for 0th crossing, but this is not set yet!!!
    //    if (gctNonIsoEm_->at(i).bx() == 0) {
      int j = i; // should be gctNonIsoEm_->at(i).capIndex(); but capIndex is not set yet!!!
      d[last] = gctNonIsoEm_->at(i).raw() & 0xff;
      last++;
      d[last] = (gctNonIsoEm_->at(i).raw()>>8) & 0xff;
      last++;
      //    }
  }

}


// Output EM Candidates packing
void GctBlockPacker::writeGctCenJetBlock(unsigned char * d, L1GctJetCandCollection* coll) {

}


// Output EM Candidates packing
void GctBlockPacker::writeGctTauJetBlock(unsigned char * d, L1GctJetCandCollection* coll) {

}


// Output EM Candidates packing
void GctBlockConverter::writeGctForJetBlock(unsigned char * d, L1GctJetCandCollection* coll) {

}


// Write a header for packing
void GctBlockPacker::writeGctHeader(unsigned char * d, unsigned id) {
  d[0] = id & 0xff;
  d[1] = 0;
  d[2] = 0;
  d[3] = 0;
}


