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

  // setup block length map
  blockLength_[0x00] = 0;
  blockLength_[0x5f] = 1;   // ConcJet: Bunch Counter Pattern Test
  blockLength_[0x68] = 4;   // ConcElec: Output to Global Trigger
  blockLength_[0x69] = 16;  // ConcElec: Sort Input
  blockLength_[0x6b] = 2;   // ConcElec: GT Serdes Loopback
  blockLength_[0x6f] = 1;   // ConcElec: Bunch Counter Pattern Test
  blockLength_[0x80] = 20;  // Leaf-U1, Elec, NegEta, Sort Input
  blockLength_[0x81] = 15;  // Leaf-U1, Elec, NegEta, Raw Input
  blockLength_[0x83] = 4;   // Leaf-U1, Elec, NegEta, Sort Output
  blockLength_[0x88] = 16;  // Leaf-U2, Elec, NegEta, Sort Input
  blockLength_[0x89] = 12;  // Leaf-U2, Elec, NegEta, Raw Input
  blockLength_[0x8b] = 4;   // Leaf-U2, Elec, NegEta, Sort Output
  blockLength_[0xc0] = 20;  // Leaf-U1, Elec, PosEta, Sort Input
  blockLength_[0xc1] = 15;  // Leaf-U1, Elec, PosEta, Raw Input
  blockLength_[0xc3] = 4;   // Leaf-U1, Elec, PosEta, Sort Output
  blockLength_[0xc8] = 16;  // Leaf-U2, Elec, PosEta, Sort Input
  blockLength_[0xc9] = 12;  // Leaf-U2, Elec, PosEta, Raw Input
  blockLength_[0xcb] = 4;   // Leaf-U2, Elec, PosEta, Sort Output

  // setup converter fn map
  //  convertFn_[0x68] = &GctBlockUnpacker::wordToGctEmCand;
  //  convertFn_[0x69] = &GctBlockUnpacker::wordToGctInterEmCand;

}

GctBlockUnpacker::~GctBlockUnpacker() { }

// recognise block ID
bool GctBlockUnpacker::validBlock(unsigned id) {
  return ( blockLength_.find(id) != blockLength_.end() );
}

// return block length in 32-bit words
unsigned GctBlockUnpacker::blockLength(unsigned id) {
  return blockLength_.find(id)->second;
}

// conversion
void GctBlockUnpacker::convertBlock(const unsigned char * data, unsigned id, unsigned nSamples) {

  if (nSamples<1) { return; }

  switch (id) {
  case (0x00):
    break;
  case (0x5f) :
    break;
  case (0x68) :
    blockToGctEmCand(data, id, nSamples);
    break;
  case (0x69) : 
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x6b) :
    break;
  case (0x6f) :
    break;
  case (0x80) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x81) :
    blockToRctEmCand(data, id, nSamples);
    blockToFibres(data, id, nSamples);
    break;
  case (0x83) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x88) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x89) :
    blockToRctEmCand(data, id, nSamples);
    blockToFibres(data, id, nSamples);
    break;
  case (0x8b) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0xc0) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0xc1) :
    blockToRctEmCand(data, id, nSamples);
    blockToFibres(data, id, nSamples);
    break;
  case (0xc3) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0xc8) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0xc9) :
    blockToRctEmCand(data, id, nSamples);
    blockToFibres(data, id, nSamples);
    break;
  case (0xcb) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  default :
    edm::LogError("GCT") << "Trying to unpack an identified block, ID=" << std::hex << id << std::endl;
    break;
  }

}


// Output EM Candidates unpacking
void GctBlockUnpacker::blockToGctEmCand(const unsigned char * d, unsigned id, unsigned nSamples) {

  LogDebug("GCT") << "Unpacking EM Cands" << std::endl;

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


  LogDebug("GCT") << "Unpacked " << gctIsoEm_->size() << " iso EM cands, "
		       << gctNonIsoEm_->size() << " non-iso EM cands" << std::endl;  

}


// Internal EM Candidates unpacking
void GctBlockUnpacker::blockToGctInternEmCand(const unsigned char * d, unsigned id, unsigned nSamples) {
  for (int i=0; i<blockLength(id)*nSamples; i=i+nSamples) {  // temporarily just take 0th time sample
    unsigned offset = i*4*nSamples;
    uint16_t w0 = d[offset]   + (d[offset+1]<<8); 
    uint16_t w1 = d[offset+2] + (d[offset+3]<<8);
    gctInternEm_->push_back( L1GctInternEmCand(w0, i > 7, id, 2*i/nSamples, 0) );
    gctInternEm_->push_back( L1GctInternEmCand(w1, i > 7, id, 2*(i/nSamples)+1, 0) );
  }
}


// Input EM Candidates unpacking
// this is the last time I deal the RCT bit assignment travesty!!!
void GctBlockUnpacker::blockToRctEmCand(const unsigned char * d, unsigned id, unsigned nSamples) {

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

  // RCT crates
  unsigned first=0;
  unsigned last=0;
  if (id==0x81) { 
    first = 4;
    last = 8;
  }
  else if (id==0x89) { 
    first = 0;
    last = 3;
  }
  else if (id==0xc1) { 
    first = 13;
    last = 17;
  }
  else if (id==0xc9) { 
    first = 9;
    last = 12;
  }

  // loop over crates
  for (int crate=first; crate<=last; crate++) {

    // read SC SFP words
    for (int cyc=0; cyc<2; cyc++) {
      for (int iSfp=0; iSfp<4; iSfp++) {

	if (iSfp==4) { sfp[cyc][iSfp] = 0; } // muon bits
	else {                              // EM candidate
	  sfp[cyc][iSfp] = *p;
	}

      }
    }

    // fill SC arrays
    srcCardRouting_.SFPtoEMU(eIsoRank, eIsoCard, eIsoRgn, eNonIsoRank, eNonIsoCard, eNonIsoRgn, MIPbits, QBits, sfp);

    // create EM cands
    for (int i=0; i<4; i++) {
      rctEm_->push_back( L1CaloEmCand( eIsoRank[i], eIsoRgn[i], eIsoCard[i], crate, true) );
    }

    for (int i=0; i<4; i++) {
      rctEm_->push_back( L1CaloEmCand( eNonIsoRank[i], eNonIsoRgn[i], eNonIsoCard[i], crate, false) );
    }
    
    // move pointer
    p = p + 12*nSamples; // just get 0th time sample for now
  }

}


// Fibre unpacking
void GctBlockUnpacker::blockToFibres(const unsigned char * d, unsigned id, unsigned nSamples) {
  for (int i=0; i<blockLength(id); i++) {
    for (int j=0; j<nSamples; j++) {
      gctFibres_->push_back( L1GctFibreWord(d[i*nSamples + j], id, i, j) );
    }
  }  
}

