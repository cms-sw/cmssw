
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


GctBlockUnpackerBase::GctBlockUnpackerBase(bool hltMode):
  rctEm_(0),
  rctCalo_(0),
  gctIsoEm_(0),
  gctNonIsoEm_(0),
  gctInternEm_(0),
  gctFibres_(0),
  gctJets_(NUM_JET_CATEGORIES),
  gctHFBitCounts_(0),
  gctHFRingEtSums_(0),
  gctEtTotal_(0),
  gctEtHad_(0),
  gctEtMiss_(0),
  gctInternJetData_(0),
  gctInternEtSums_(0),
  hltMode_(hltMode),
  srcCardRouting_()
{
}

GctBlockUnpackerBase::~GctBlockUnpackerBase() { }

// conversion
bool GctBlockUnpackerBase::checkBlock(const GctBlockHeaderBase& hdr)
{
  // check block is valid
  if ( !hdr.valid() )
  {
    LogDebug("GCT") << "Block unpack error: cannot unpack the following unknown/invalid block:\n" << hdr;
    return false;     
  }

  // check block doesn't have too many time samples
  if ( hdr.nSamples() >= 0xf ) {
    LogDebug("GCT") << "Block unpack error: cannot unpack a block with 15 or more time samples:\n" << hdr;
    return false; 
  }
  return true;
}

// Internal EM Candidates unpacking
void GctBlockUnpackerBase::blockToGctInternEmCand(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal EM Cands"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int numCandPairs = hdr.length();

  // Debug assertion to prevent problems if definitions not up to date.
  assert(internEmIsoBounds().find(id) != internEmIsoBounds().end());  

  unsigned int lowerIsoPairBound = internEmIsoBounds()[id].first;
  unsigned int upperIsoPairBound = internEmIsoBounds()[id].second;

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
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT EM Cands"; return; }

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
  for (unsigned int crate=rctCrateMap()[id]; crate<rctCrateMap()[id]+length/3; ++crate) {

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

// Input RCT region unpacking
void GctBlockUnpackerBase::blockToRctCaloRegions(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT Regions"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Debug assertion to prevent problems if definitions not up to date.
  assert(rctJetCrateMap().find(id) != rctJetCrateMap().end());  
  
  // get crate (need this to get ieta and iphi)
  unsigned int crate=rctJetCrateMap()[id];

  // re-interpret pointer
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));
  
  // eta and phi
  unsigned int ieta; 
  unsigned int iphi; 
  
  for (unsigned int i=0; i<length; ++i) { 
    for (uint16_t bx=0; bx<nSamples; ++bx) {
      if (i>0) {
        if (crate<9){ // negative eta
          ieta = 11-i; 
          iphi = 2*((11-crate)%9);
        } else {      // positive eta
          ieta = 10+i;
          iphi = 2*((20-crate)%9);
        }        
        // First region is phi=0
        rctCalo_->push_back( L1CaloRegion::makeRegionFromUnpacker(*p, ieta, iphi, id, i, bx) );
        ++p;
        // Second region is phi=1
        if (iphi>0){
          iphi-=1;
        } else {
          iphi = 17;
        }
        rctCalo_->push_back( L1CaloRegion::makeRegionFromUnpacker(*p, ieta, iphi, id, i, bx) );
        ++p;
      } else { // Skip the first two regions which are duplicates. 
        ++p;
        ++p;
      }
    }
  } 
}  

// Fibre unpacking
void GctBlockUnpackerBase::blockToFibres(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of GCT Fibres"; return; }
  
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

void GctBlockUnpackerBase::blockToAllRctCaloRegions(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT Calo Regions"; return; }

  // This method is one giant "temporary" hack whilst waiting for proper
  // pipeline formats for the RCT calo region data.
  
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

void GctBlockUnpackerBase::blockToGctInternEtSums(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!                                                                                                                           
  
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Et Sums"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 32 bits 
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)                                                                                                                            
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      gctInternEtSums_->push_back(L1GctInternEtSum::fromTotalEtOrHt(id,i,bx,*p));
      ++p;
    }
  }
}

void GctBlockUnpackerBase::blockToGctInternEtSumsAndJetCluster(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 32 bits 
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      if (i<2) gctInternEtSums_->push_back(L1GctInternEtSum::fromJetMissEt(id,i,bx,*p));
      if (i==3){
        gctInternEtSums_->push_back(L1GctInternEtSum::fromJetTotEt(id,i,bx,*p));
        gctInternEtSums_->push_back(L1GctInternEtSum::fromJetTotHt(id,i,bx,*p));
      } 
      if (i>4) gctInternJetData_->push_back(L1GctInternJetData::fromJetCluster(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }  
  }
}

void GctBlockUnpackerBase::blockToGctTrigObjects(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      gctInternJetData_->push_back( L1GctInternJetData::fromGctTrigObject(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
      gctInternJetData_->push_back( L1GctInternJetData::fromGctTrigObject(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }
  } 
}

void GctBlockUnpackerBase::blockToGctJetClusterMinimal(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      gctInternJetData_->push_back( L1GctInternJetData::fromJetClusterMinimal(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
      gctInternJetData_->push_back( L1GctInternJetData::fromJetClusterMinimal(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }
  } 
}

void GctBlockUnpackerBase::blockToGctJetPreCluster(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal Jet Cands"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 16 bits so it sees one candidate at a time.
  uint16_t * p = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      gctInternJetData_->push_back( L1GctInternJetData::fromJetPreCluster(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
      gctInternJetData_->push_back( L1GctInternJetData::fromJetPreCluster(L1CaloRegionDetId(0,0),id,i,bx,*p));
      ++p;
    }
  } 
}

void GctBlockUnpackerBase::blockToGctInternRingSums(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of internal HF ring data"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 32 bits 
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length/2; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      gctInternHFData_->push_back(L1GctInternHFData::fromConcRingSums(id,i,bx,*p));
      ++p;
    }
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      gctInternHFData_->push_back(L1GctInternHFData::fromConcBitCounts(id,i,bx,*p));
      ++p;
    }  
  }
}

void GctBlockUnpackerBase::blockToGctWheelInputInternEtAndRingSums(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of wheel input internal Et sums and HF ring data"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 32 bits 
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      if (i<3){
        gctInternEtSums_->push_back(L1GctInternEtSum::fromTotalEtOrHt(id,i,bx,*p));
      } else if (i>2 && i<9) {
        gctInternEtSums_->push_back(L1GctInternEtSum::fromMissEtxOrEty(id,i,bx,*p));
      } else if (i>8 && i<15) {
        gctInternHFData_->push_back(L1GctInternHFData::fromWheelRingSums(id,i,bx,*p));
      } else if (i>14){
        gctInternHFData_->push_back(L1GctInternHFData::fromWheelBitCounts(id,i,bx,*p));
      }
      ++p;
    }
  }
}

void GctBlockUnpackerBase::blockToGctWheelOutputInternEtAndRingSums(const unsigned char * d, const GctBlockHeaderBase& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of wheel output internal Et sums and HF ring data"; return; }

  unsigned int id = hdr.id();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.length();

  // Re-interpret pointer to 32 bits 
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    // Loop over timesamples (i.e. bunch crossings)
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      if (i<1){
        gctInternEtSums_->push_back(L1GctInternEtSum::fromTotalEtOrHt(id,i,bx,*p));
      } else if (i>0 && i<3) {
        gctInternEtSums_->push_back(L1GctInternEtSum::fromMissEtxOrEty(id,i,bx,*p));
      } else if (i>2 && i<5) {
        gctInternHFData_->push_back(L1GctInternHFData::fromWheelRingSums(id,i,bx,*p));
      } else if (i>4){
        gctInternHFData_->push_back(L1GctInternHFData::fromWheelBitCounts(id,i,bx,*p));
      }
      ++p;
    }
  }
}

