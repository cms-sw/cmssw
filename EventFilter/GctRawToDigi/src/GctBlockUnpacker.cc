#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/GctRawToDigi/src/GctBlockUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <iostream>

using std::cout;
using std::endl;

// INITIALISE STATIC VARIABLES
GctBlockUnpacker::RctCrateMap GctBlockUnpacker::rctCrate_ = GctBlockUnpacker::RctCrateMap();
GctBlockUnpacker::BlockIdToUnpackFnMap GctBlockUnpacker::blockUnpackFn_ = GctBlockUnpacker::BlockIdToUnpackFnMap();


GctBlockUnpacker::GctBlockUnpacker() :
  rctEm_(0),
  gctIsoEm_(0),
  gctNonIsoEm_(0),
  gctInternEm_(0),
  gctFibres_(0)
{
  static bool initClass = true;
  
  if(initClass)
  {
    initClass = false;
    
    // RCT crates
    rctCrate_[0x81] = 4;
    rctCrate_[0x89] = 0;
    rctCrate_[0xC1] = 13;
    rctCrate_[0xC9] = 9;

    // Setup block unpack function map.
    blockUnpackFn_[0x00] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x58] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x59] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x5f] = &GctBlockUnpacker::blockDoNothing;
    blockUnpackFn_[0x68] = &GctBlockUnpacker::blockToGctEmCand;
    blockUnpackFn_[0x69] = &GctBlockUnpacker::blockToGctInternEmCand;
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
  }

}

GctBlockUnpacker::~GctBlockUnpacker() { }

// conversion
void GctBlockUnpacker::convertBlock(const unsigned char * data, GctBlockHeader& hdr) {

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();

  // if the block has no time samples, don't bother
  if ( nSamples < 1 ) { return; }

  // check block is valid
  if ( !hdr.valid() ) {
    edm::LogError("GCT") << "Attempting to unpack an unidentified block\n" << hdr << endl;
    return;     
  }

  // check block doesn't have too many time samples
  if ( nSamples >= 0xf ) {
    edm::LogError("GCT") << "Cannot unpack a block with 15 or more time samples\n" << hdr << endl;
    return; 
  }

  // The header validity check above will protect against 
  // the map::find() method returning the end of the map,
  // assuming the GctBlockHeader definitions are up-to-date.
  (this->*blockUnpackFn_.find(id)->second)(data, hdr);

}


// Output EM Candidates unpacking
void GctBlockUnpacker::blockToGctEmCand(const unsigned char * d, const GctBlockHeader& hdr) {

  LogDebug("GCT") << "Unpacking GCT output EM Cands" << std::endl;

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();

  // Re-interpret pointer.  p will be pointing at the 16 bit word that
  // contains the rank0 non-isolated electron of the zeroth time-sample. 
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

  for (unsigned int iso=0; iso<2; ++iso) {   // loop over non-iso/iso candidates
    for (unsigned int bx=0; bx<nSamples; ++bx) {   // loop over time samples

      bool isolated = (iso==1);
      
      // The +(2*bx) for the start of the correct time sample
      // The +(iso*4*nSamples) for selecting the start of the non-iso/iso. 
      uint16_t * pp = p + (2*bx) + (iso*4*nSamples);
      

      L1GctEmCandCollection* em;
      if (isolated) { em = gctIsoEm_; }
      else { em = gctNonIsoEm_; }

      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));  // rank0 electron
      pp = pp + (2*(nSamples-1)) + 2;
      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));  // rank1 electron
      pp = pp - (2*(nSamples-1)) - 1;
      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));  // rank2 electron
      pp = pp + (2*(nSamples-1)) + 2;
      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));  // rank3 electron

    }
  }

}


// Internal EM Candidates unpacking
void GctBlockUnpacker::blockToGctInternEmCand(const unsigned char * d, const GctBlockHeader& hdr) {

  LogDebug("GCT") << "Unpacking internal EM Cands" << std::endl;

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  for (unsigned int i=0 ; i<length*nSamples ; i+=nSamples) {  // temporarily just take 0th time sample
    unsigned offset = i*4*nSamples;
    uint16_t w0 = d[offset]   + (d[offset+1]<<8); 
    uint16_t w1 = d[offset+2] + (d[offset+3]<<8);
    gctInternEm_->push_back( L1GctInternEmCand(w0, i > 7, id, 2*i/nSamples, 0) );
    gctInternEm_->push_back( L1GctInternEmCand(w1, i > 7, id, 2*(i/nSamples)+1, 0) );
  }

}


// Input EM Candidates unpacking
// this is the last time I deal the RCT bit assignment travesty!!!
void GctBlockUnpacker::blockToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr) {

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
  for (unsigned int crate=rctCrate_[id]; crate<length/3; ++crate) {

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
void GctBlockUnpacker::blockToFibres(const unsigned char * d, const GctBlockHeader& hdr) {

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

void GctBlockUnpacker::blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr) {
  this->blockToFibres(d, hdr);
  this->blockToRctEmCand(d, hdr);
}

