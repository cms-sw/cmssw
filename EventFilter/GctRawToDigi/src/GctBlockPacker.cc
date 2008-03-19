#include "EventFilter/GctRawToDigi/src/GctBlockPacker.h"

// C++ headers
#include <cassert>
#include <iostream>

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// GctRawToDigi headers
#include "EventFilter/GctRawToDigi/src/GctBlockHeaderV2.h"

// Namespace resolution
using std::endl;
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
    
    rctCrate_[0x804] = 13;
    rctCrate_[0x884] = 9;
    rctCrate_[0xc04] = 4;
    rctCrate_[0xc84] = 0; 
  }  
}

GctBlockPacker::~GctBlockPacker() {

}


// write GCT internal header
void GctBlockPacker::writeGctHeader(unsigned char * d, uint16_t id, uint16_t nsamples)
{
  uint32_t hdr = GctBlockHeaderV2(id, nsamples, bcid_, evid_).data();
  uint32_t * p = reinterpret_cast<uint32_t*>(const_cast<unsigned char *>(d));
  *p = hdr;
}


void GctBlockPacker::writeGctOutJetBlock(unsigned char * d,
                                         const L1GctJetCandCollection* cenJets,
                                         const L1GctJetCandCollection* forJets,
                                         const L1GctJetCandCollection* tauJets, 
                                         const L1GctJetCounts* jetCounts)
{
  if(cenJets->size()<4 || forJets->size()<4 || tauJets->size()<4)  // Simple guard clause to stop crappy data from crashing the packer.
  {
    edm::LogError("GCT") << "Block pack error: bad L1GctJetCandCollection size detected!\n"
                         << "L1GctJetCandCollecion size is too small for at least one of the collections!\n"
                         << "Aborting packing of GCT Jet Output to GT data!" << endl;
    return;
  }
  
  // write header
  writeGctHeader(d, 0x583, 1);
  
  d=d+4;  // move forward past the block header to the block payload.

  // FIRST DO JET CANDS
  // re-interpret pointer to 16 bits - the space allocated for each Jet candidate.
  uint16_t * p16 = reinterpret_cast<uint16_t *>(d);
  
  // Set up a vector of the collections for easy iteration.
  vector<const L1GctJetCandCollection*> jets(NUM_JET_CATEGORIES);
  jets.at(CENTRAL_JETS)=cenJets;
  jets.at(FORWARD_JETS)=forJets;
  jets.at(TAU_JETS)=tauJets;

  const unsigned int categoryOffset = 4;  // Offset to jump from one jet category to the next.
  const unsigned int nextCandPairOffset = 2;  // Offset to jump to next candidate pair.

  // Loop over the different catagories of jets
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATEGORIES ; ++iCat)
  {
    // cand0Offset will give the offset on p16 to get the rank 0 Jet Cand of the correct category.
    const unsigned int cand0Offset = iCat*categoryOffset;
    
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

// Output EM Candidates and energy sums packing
void GctBlockPacker::writeGctOutEmAndEnergyBlock(unsigned char * d,
                                                 const L1GctEmCandCollection* iso,
                                                 const L1GctEmCandCollection* nonIso,
                                                 const L1GctEtTotal* etTotal,
                                                 const L1GctEtHad* etHad,
                                                 const L1GctEtMiss* etMiss)
{
  if(iso->size()<4 || nonIso->size()<4)  // Simple guard clause to stop crappy data from crashing the packer.
  {
    edm::LogError("GCT") << "Block pack error: bad L1GctEmCandCollection size detected!\n"
                         << "L1GctEmCandCollection size is too small for at least one of the collections!\n"
                         << "Aborting packing of GCT EM Cands + EnergySum Output to GT data!" << endl;
    return;
  }

  unsigned nSamples = 1; // Note: can only currently do SINGLE TIME SAMPLE!
  
  // write header
  writeGctHeader(d, 0x683, nSamples);  
  
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

void GctBlockPacker::writeRctEmCandBlocks(unsigned char * d, const L1CaloEmCollection * rctEm)
{
  // This method is one giant "temporary" hack for CMSSW_1_8_X and CMSSW_2_0_0.

  if(rctEm->size() == 0 || rctEm->size()%144 != 0)  // Should be 18 crates * 2 types (iso/noniso) * 4 electrons = 144 for 1 bx.
  {
    edm::LogError("GCT") << "Block pack error: bad L1CaloEmCollection size detected!\n"
                         << "Aborting packing of RCT EM Cand data!" << endl;
    return;
  }

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
    unsigned blockLength_32bit = GctBlockHeaderV2(blockId, 1, 0, 0).length();
    
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
  // This method is one giant "temporary" hack for CMSSW_1_8_X and CMSSW_2_0_0.

  if(rctEm->size() == 0 || rctCalo->size()%396 != 0)  // Should be 396 calo regions for 1 bx.
  {
    edm::LogError("GCT") << "Block pack error: bad L1CaloRegionCollection size detected!\n"
                         << "Aborting packing of RCT Calo Region data!" << endl;
    return;

  writeGctHeader(d, 0x0ff, 1);
  d+=4; // move past header.

  // Want a 16 bit pointer to push the 16 bit data in.
  uint16_t * p16 = reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(d));
 
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
