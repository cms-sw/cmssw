#include "EventFilter/GctRawToDigi/src/GctBlockPacker.h"

// C++ headers
#include <cassert>

// CMSSW headers
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

// Namespace resolution
using std::vector;

// INITIALISE STATICS
GctBlockPacker::RctCrateMap GctBlockPacker::rctCrate_ = GctBlockPacker::RctCrateMap();


GctBlockPacker::GctBlockPacker():
  bcid_(0),
  evid_(0),
  srcCardRouting_()
{
  static bool initClass = true;
  
  if(initClass)
  {
    initClass = false;
    
    rctCrate_[0x81] = 13;
    rctCrate_[0x89] = 9;
    rctCrate_[0xC1] = 4;
    rctCrate_[0xC9] = 0; 
  }  
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


void GctBlockPacker::writeGctJetCandsBlock(unsigned char * d,
                                           const L1GctJetCandCollection* cenJets,
                                           const L1GctJetCandCollection* forJets,
                                           const L1GctJetCandCollection* tauJets)
{
  // Set up a vector of the collections for easy iteration.
  vector<const L1GctJetCandCollection*> jets(NUM_JET_CATAGORIES);
  jets.at(CENTRAL_JETS)=cenJets;
  jets.at(FORWARD_JETS)=forJets;
  jets.at(TAU_JETS)=tauJets;
  
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
      p[cand0Offset + timeSampleOffset] = jets.at(iCat)->at(1).raw(); // rank 1 jet
      p[cand0Offset + 1] = jets.at(iCat)->at(2).raw(); // rank 2 jet
      p[cand0Offset + timeSampleOffset + 1] = jets.at(iCat)->at(3).raw(); // rank 3 jet.
    }
  }
}


void GctBlockPacker::writeGctJetCountsBlock(unsigned char * d, const L1GctJetCounts* jetCounts)
{
  writeGctHeader(d, 0x5A, 1);  // Write header - timesamples set to 1.
  
  d=d+4; // Move to block payload.
  
  // re-interpret pointer to 32 bit.
  uint32_t * p = reinterpret_cast<uint32_t *>(d);
  
  p[0] = jetCounts->raw0();
  p[1] = jetCounts->raw1();  
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

  // re-interpret payload pointer to 16 bit.
  uint16_t * p = reinterpret_cast<uint16_t *>(d);

  for (int i=0; i<2; i++)   // loop over non-iso/iso candidates
  {
    for (int bx=0; bx<nSamples; bx++)   // loop over time samples
    {
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


// Output Energy Sums packing
void GctBlockPacker::writeGctEnergySumsBlock(unsigned char * d, const L1GctEtTotal* etTotal,
                                             const L1GctEtHad* etHad, const L1GctEtMiss* etMiss)
{
  writeGctHeader(d, 0x6A, 1);  // Write header - timesamples set to 1.
  
  d=d+4; // Move to block payload.
  
  // Re-interpret block payload pointer to both 16 and 32 bits.  
  uint16_t * p16 = reinterpret_cast<uint16_t *>(d);  // For writing Et + Ht (16-bit raw data each)
  uint32_t * p32 = reinterpret_cast<uint32_t *>(d);  // For writing Missing Et (32-bit raw data)
  
  p16[0] = etTotal->raw();  // Et Total on bits 15:0 of block payload.
  p16[1] = etHad->raw();  // Et Hadronic on bits 31:16 of block payload.
  p32[1] = etMiss->raw();  // Et Miss on bits 63:32 of block payload.
}

void GctBlockPacker::writeRctEmCandBlocks(unsigned char * d, const L1CaloEmCollection * rctEm)
{
  // This method is one giant "temporary" hack for CMSSW_1_8_X.

  assert(rctEm->size() >= 144);  // Should be 18 crates * 2 types (iso/noniso) * 4 electrons = 144 for 1 bx.

  // Need 18 sets of EM fibre data, since 18 RCT crates  
  EmuToSfpData emuToSfpData[18];

  // Fill in the input arrays with the data from the digi  
  for(unsigned i=0, size=rctEm->size(); i < size ; ++i)
  {
    const L1CaloEmCand &cand = rctEm->at(i);
    if(cand.bx() != 0) { continue; }  // Only interested in bunch crossing zero for now!
    unsigned crateNum = cand.rctCrate();
    unsigned index = cand.index();
    
    // Some error checking.
    assert(crateNum < 18); // Only 18 RCT crates!
    assert(index < 4); // Should only be 4 cands of each type per crate!
    
    if(cand.isolated())
    {
      emuToSfpData[crateNum].eIsoRank[index] = cand.rank();
      emuToSfpData[crateNum].eIsoCardId[index] = cand.rctCard();
      emuToSfpData[crateNum].eIsoRegionId[index] = cand.rctRegion();
    }
    else
    {
      emuToSfpData[crateNum].eNonIsoRank[index] = cand.rank();
      emuToSfpData[crateNum].eNonIsoCardId[index] = cand.rctCard();
      emuToSfpData[crateNum].eNonIsoRegionId[index] = cand.rctRegion();
    }
    // Note doing nothing with the MIP bit and Q bit arrays as we are not
    // interested in them; these arrays will contain uninitialised junk
    // and so you will get out junk for sourcecard output 0 - I.e. don't
    // trust sfp[0][0] or sfp[1][0] output!. 
  }

  // Now run the conversion
  for(unsigned c = 0 ; c < 18 ; ++c)
  {
    srcCardRouting_.EMUtoSFP(emuToSfpData[c].eIsoRank, emuToSfpData[c].eIsoCardId, emuToSfpData[c].eIsoRegionId,
                             emuToSfpData[c].eNonIsoRank, emuToSfpData[c].eNonIsoCardId, emuToSfpData[c].eNonIsoRegionId,
                             emuToSfpData[c].mipBits, emuToSfpData[c].qBits, emuToSfpData[c].sfp);
  }
  
  // Now pack up the data into the RAW format.
  RctCrateMap::iterator blockStartCrateIter;
  for(blockStartCrateIter = rctCrate_.begin() ; blockStartCrateIter != rctCrate_.end() ; ++blockStartCrateIter)
  {
    unsigned blockId = blockStartCrateIter->first;
    unsigned startCrate = blockStartCrateIter->second;
    unsigned blockLength_32bit = GctBlockHeader::lookupBlockLength(blockId);
    
    writeGctHeader(d, blockId, 1);
    d+=4; // move past header.
    
    // Want a 16 bit pointer to push the 16 bit data in.
    uint16_t * p16 = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));
    
    for(unsigned iCrate=startCrate, end=startCrate + blockLength_32bit/3 ; iCrate < end ; ++iCrate)
    {
      for(unsigned iOutput = 1 ; iOutput < 4 ; ++iOutput)  // skipping output 0 as that is Q-bit/MIP-bit data.
      {
        for(unsigned iCycle = 0 ; iCycle < 2 ; ++iCycle)
        {
          *p16 = emuToSfpData[iCrate].sfp[iCycle][iOutput];
          ++p16;
        }
      } 
    }
    
    // Now move d onto the location of the next block header
    d+=(blockLength_32bit*4);
  }
}

void GctBlockPacker::writeRctCaloRegionBlock(unsigned char * d, const L1CaloRegionCollection * rctCalo)
{
  // This method is one giant "temporary" hack for CMSSW_1_8_X.

  writeGctHeader(d, 0xff, 1);
  d+=4; // move past header.

  // Want a 16 bit pointer to push the 16 bit data in.
  uint16_t * p16 = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));
 
  assert(rctCalo->size() >= 396);  // Should be at least 396 calo regions for 1 bx.
  
  for(unsigned i=0, size=rctCalo->size(); i < size ; ++i)
  {
    const L1CaloRegion &reg = rctCalo->at(i);
    if(reg.bx() != 0) { continue; }  // Only interested in bunch crossing zero for now!
    const unsigned crateNum = reg.rctCrate();
    const unsigned regionIndex = reg.rctRegionIndex();
    assert(crateNum < 18); // Only 18 RCT crates!
    
    // Gotta make the raw data as there currently isn't a method of getting raw from L1CaloRegion
    const uint16_t raw =  reg.et()                        | 
                         (reg.overFlow()  ? 0x400  : 0x0) |
                         (reg.fineGrain() ? 0x800  : 0x0) |
                         (reg.mip()       ? 0x1000 : 0x0) |
                         (reg.quiet()     ? 0x2000 : 0x0);
 
    unsigned offset = 0;  // for storing calculated raw data offset.   
    if(reg.isHbHe())  // Is a barrel/endcap region
    {
      const unsigned cardNum = reg.rctCard();
      assert(cardNum < 7);  // 7 RCT cards per crate for the barrel/endcap
      assert(regionIndex < 2); // regionIndex less than 2 for barrel/endcap
      
      // Calculate position in the raw data from crateNum, cardNum, and regionIndex
      offset = crateNum*22 + cardNum*2 + regionIndex;
    }
    else  // Must be forward region
    {
      assert(regionIndex < 8); // regionIndex less than 8 for forward calorimeter.
      offset = crateNum*22 + 14 + regionIndex;
    }
    p16[offset] = raw;  // Write raw data in correct place!
  }
}
