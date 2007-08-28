#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/GctRawToDigi/src/GctBlockUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <iostream>

//#define CALL_GCT_CONVERT_FN(object,ptrToMember)  ((object).*(ptrToMember))

using std::cout;
using std::endl;

GctBlockUnpacker::GctBlockUnpacker() :
  rctEm_(0),
  gctIsoEm_(0),
  gctNonIsoEm_(0),
  gctInternEm_(0),
  gctFibres_(0)
{

  // RCT crates
  unsigned first=0;
  //  unsigned last=0;
  rctCrate_[0x81] = 4;
  rctCrate_[0x89] = 0;
  rctCrate_[0xC1] = 13;
  rctCrate_[0xC9] = 9;

  // setup converter fn map
//   convertFn_[0x68] = &GctBlockUnpacker::blockToGctEmCand;
//   convertFn_[0x69] = &GctBlockUnpacker::blockToGctInternEmCand;
//   convertFn_[0x81] = &GctBlockUnpacker::blockToRctEmCand;
//   convertFn_[0x83] = &GctBlockUnpacker::blockToGctInternEmCand;
//   convertFn_[0x88] = &GctBlockUnpacker::blockToGctInternEmCand;
//   convertFn_[0x89] = &GctBlockUnpacker::blockToRctEmCand;
//   convertFn_[0x8b] = &GctBlockUnpacker::blockToGctInternEmCand;
//   convertFn_[0xc0] = &GctBlockUnpacker::blockToGctInternEmCand;
//   convertFn_[0xc1] = &GctBlockUnpacker::blockToRctEmCand;
//   convertFn_[0xc3] = &GctBlockUnpacker::blockToGctInternEmCand;
//   convertFn_[0xc8] = &GctBlockUnpacker::blockToGctInternEmCand;
//   convertFn_[0xc9] = &GctBlockUnpacker::blockToRctEmCand;
//   convertFn_[0xcb] = &GctBlockUnpacker::blockToGctInternEmCand;

}

GctBlockUnpacker::~GctBlockUnpacker() { }

// conversion
void GctBlockUnpacker::convertBlock(const unsigned char * data, GctBlockHeader& hdr) {

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();

  // if the block has no time samples, don't bother
  if ( nSamples > 1 ) { return; }

  // check block is valid
  if ( hdr.valid() ) {
    std::ostringstream os;
    os << "Attempting to unpack an unidentified block" << std::endl; 
    os << hdr << endl;
    edm::LogError("GCT") << os.str();
    return;     
  }

  // check block doesn't have too many time samples
  if ( nSamples >= 0xf ) {
    std::ostringstream os;
    os << "Cannot unpack a block with 15 or more time samples" << std::endl; 
    os << hdr << endl;
    edm::LogError("GCT") << os.str();
    return; 
  }

  // if this is a fibre block unpack fibres
  if (id==0x81 || id==0x89 || id==0xc1 || id==0xc9) {
    this->blockToFibres(data, hdr);
  }
  
  // unpack the block
  //  CALL_GCT_CONVERT_FN((*this), convertFn_.find(id).second)(data, hdr);

  switch (hdr.id()) {
  case (0x00):
    break;
  case (0x58):
    break;
  case (0x59):
    break;
  case (0x5f) :
    break;
  case (0x68) :
    blockToGctEmCand(data, hdr);
    break;
  case (0x69) : 
    blockToGctInternEmCand(data, hdr);
    break;
  case (0x6b) :
    break;
  case (0x6f) :
    break;
  case (0x80) :
    blockToGctInternEmCand(data, hdr);
    break;
  case (0x81) :
    blockToRctEmCand(data, hdr);
    break;
  case (0x83) :
    blockToGctInternEmCand(data, hdr);
    break;
  case (0x88) :
    blockToGctInternEmCand(data, hdr);
    break;
  case (0x89) :
    blockToRctEmCand(data, hdr);
    break;
  case (0x8b) :
    blockToGctInternEmCand(data, hdr);
    break;
  case (0xc0) :
    blockToGctInternEmCand(data, hdr);
    break;
  case (0xc1) :
    blockToRctEmCand(data, hdr);
    break;
  case (0xc3) :
    blockToGctInternEmCand(data, hdr);
    break;
  case (0xc8) :
    blockToGctInternEmCand(data, hdr);
    break;
  case (0xc9) :
    blockToRctEmCand(data, hdr);
    break;
  case (0xcb) :
    blockToGctInternEmCand(data, hdr);
    break;
  default :
    edm::LogError("GCT") << "Trying to unpack an identified block, ID=" << std::hex << hdr.id() << std::endl;
    break;
  }

}


// Output EM Candidates unpacking
void GctBlockUnpacker::blockToGctEmCand(const unsigned char * d, GctBlockHeader& hdr) {

  LogDebug("GCT") << "Unpacking GCT output EM Cands" << std::endl;

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // re-interpret pointer
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

  for (int iso=0; iso<2; iso++) {   // loop over non-iso/iso candidates
    for (int bx=0; bx<nSamples; bx++) {   // loop over time samples

      bool isolated = (iso==1);
      uint16_t * pp = p+ (2*bx) + (iso*4*nSamples);

      L1GctEmCandCollection* em;
      if (isolated) { em = gctIsoEm_; }
      else { em = gctNonIsoEm_; }

      pp=pp+3+(2*(nSamples-1));
      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));
      pp=pp-(2*(nSamples-1))-2;
      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));
      pp=pp+(2*(nSamples-1))+1;
      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));
      pp=pp-(2*(nSamples-1))-2;
      em->push_back(L1GctEmCand(*pp, isolated, id, 0, bx));

    }
  }

}


// Internal EM Candidates unpacking
void GctBlockUnpacker::blockToGctInternEmCand(const unsigned char * d, GctBlockHeader& hdr) {

  LogDebug("GCT") << "Unpacking internal EM Cands" << std::endl;

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  for (int i=0; i<length*nSamples; i=i+nSamples) {  // temporarily just take 0th time sample
    unsigned offset = i*4*nSamples;
    uint16_t w0 = d[offset]   + (d[offset+1]<<8); 
    uint16_t w1 = d[offset+2] + (d[offset+3]<<8);
    gctInternEm_->push_back( L1GctInternEmCand(w0, i > 7, id, 2*i/nSamples, 0) );
    gctInternEm_->push_back( L1GctInternEmCand(w1, i > 7, id, 2*(i/nSamples)+1, 0) );
  }

}


// Input EM Candidates unpacking
// this is the last time I deal the RCT bit assignment travesty!!!
void GctBlockUnpacker::blockToRctEmCand(const unsigned char * d, GctBlockHeader& hdr) {

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

  int bx = 0;

  // loop over crates
  for (int crate=rctCrate_[id]; crate<length/3; crate++) {

    // read SC SFP words
    for (int iSfp=0; iSfp<4; iSfp++) {
      for (int cyc=0; cyc<2; cyc++) {
	
	if (iSfp==0) { sfp[cyc][iSfp] = 0; } // muon bits
	else {                               // EM candidate
	  sfp[cyc][iSfp] = *p;
	  p++;
	}
	  
      }
      
      p = p + 2*(nSamples-1);
    }

    // fill SC arrays
    srcCardRouting_.SFPtoEMU(eIsoRank, eIsoCard, eIsoRgn, eNonIsoRank, eNonIsoCard, eNonIsoRgn, MIPbits, QBits, sfp);
    
    // create EM cands
    for (int i=0; i<4; i++) {
      rctEm_->push_back( L1CaloEmCand( eIsoRank[i], eIsoRgn[i], eIsoCard[i], crate, true, i, bx) );
    }
    for (int i=0; i<4; i++) {
      rctEm_->push_back( L1CaloEmCand( eNonIsoRank[i], eNonIsoRgn[i], eNonIsoCard[i], crate, false, i, bx) );
    }
    
  }
  
}


// Fibre unpacking
void GctBlockUnpacker::blockToFibres(const unsigned char * d, GctBlockHeader& hdr) {

  LogDebug("GCT") << "Unpacking GCT Fibres" << std::endl;
  
  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // re-interpret pointer
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (int i=0; i<length; i++) {
    for (int bx=0; bx<nSamples; bx++) {
      gctFibres_->push_back( L1GctFibreWord(*p, id, i, bx) );
      p++;
    }
  } 
 
}

