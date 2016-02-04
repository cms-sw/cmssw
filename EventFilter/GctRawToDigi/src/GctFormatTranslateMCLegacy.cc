#include "EventFilter/GctRawToDigi/src/GctFormatTranslateMCLegacy.h"

// C++ headers
#include <iostream>
#include <cassert>

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Namespace resolution
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;
using std::vector;

// INITIALISE STATIC VARIABLES
GctFormatTranslateMCLegacy::BlockLengthMap GctFormatTranslateMCLegacy::m_blockLength = GctFormatTranslateMCLegacy::BlockLengthMap();
GctFormatTranslateMCLegacy::BlockNameMap GctFormatTranslateMCLegacy::m_blockName = GctFormatTranslateMCLegacy::BlockNameMap();
GctFormatTranslateMCLegacy::BlockIdToUnpackFnMap GctFormatTranslateMCLegacy::m_blockUnpackFn = GctFormatTranslateMCLegacy::BlockIdToUnpackFnMap();
GctFormatTranslateMCLegacy::BlkToRctCrateMap GctFormatTranslateMCLegacy::m_rctEmCrate = GctFormatTranslateMCLegacy::BlkToRctCrateMap();
GctFormatTranslateMCLegacy::BlkToRctCrateMap GctFormatTranslateMCLegacy::m_rctJetCrate = GctFormatTranslateMCLegacy::BlkToRctCrateMap();
GctFormatTranslateMCLegacy::BlockIdToEmCandIsoBoundMap GctFormatTranslateMCLegacy::m_internEmIsoBounds = GctFormatTranslateMCLegacy::BlockIdToEmCandIsoBoundMap();


// PUBLIC METHODS

GctFormatTranslateMCLegacy::GctFormatTranslateMCLegacy(bool hltMode, bool unpackSharedRegions):
  GctFormatTranslateBase(hltMode, unpackSharedRegions)
{
  static bool initClass = true;

  if(initClass)
  {
    initClass = false;

    /*** Setup BlockID to BlockLength Map ***/
    // Miscellaneous Blocks
    m_blockLength.insert(make_pair(0x000,0));      // NULL
    m_blockLength.insert(make_pair(0x0ff,198));    // Temporary hack: All RCT Calo Regions for CMSSW pack/unpack
    // ConcJet FPGA
    m_blockLength.insert(make_pair(0x583,8));      // ConcJet: Jet Cands and Counts Output to GT
    // ConcElec FPGA
    m_blockLength.insert(make_pair(0x683,6));      // ConcElec: EM Cands and Energy Sums Output to GT
    // Electron Leaf FPGAs
    m_blockLength.insert(make_pair(0x804,15));     // Leaf0ElecPosEtaU1: Raw Input
    m_blockLength.insert(make_pair(0x884,12));     // Leaf0ElecPosEtaU2: Raw Input
    m_blockLength.insert(make_pair(0xc04,15));     // Leaf0ElecNegEtaU1: Raw Input
    m_blockLength.insert(make_pair(0xc84,12));     // Leaf0ElecNegEtaU2: Raw Input


    /*** Setup BlockID to BlockName Map ***/
    // Miscellaneous Blocks
    m_blockName.insert(make_pair(0x000,"NULL"));
    m_blockName.insert(make_pair(0x0ff,"All RCT Calo Regions"));  // Temporary hack: All RCT Calo Regions for CMSSW pack/unpack
    // ConcJet FPGA
    m_blockName.insert(make_pair(0x583,"ConcJet: Jet Cands and Counts Output to GT"));
    // ConcElec FPGA
    m_blockName.insert(make_pair(0x683,"ConcElec: EM Cands and Energy Sums Output to GT"));
    // Electron Leaf FPGAs
    m_blockName.insert(make_pair(0x804,"Leaf0ElecPosEtaU1: Raw Input"));
    m_blockName.insert(make_pair(0x884,"Leaf0ElecPosEtaU2: Raw Input"));
    m_blockName.insert(make_pair(0xc04,"Leaf0ElecNegEtaU1: Raw Input"));
    m_blockName.insert(make_pair(0xc84,"Leaf0ElecNegEtaU2: Raw Input"));


    /*** Setup BlockID to Unpack-Function Map ***/
    // Miscellaneous Blocks
    m_blockUnpackFn[0x000] = &GctFormatTranslateMCLegacy::blockDoNothing;                    // NULL
    m_blockUnpackFn[0x0ff] = &GctFormatTranslateMCLegacy::blockToAllRctCaloRegions;          // Temporary hack: All RCT Calo Regions for CMSSW pack/unpack
    // ConcJet FPGA                                                             
    m_blockUnpackFn[0x583] = &GctFormatTranslateMCLegacy::blockToGctJetCandsAndCounts;       // ConcJet: Jet Cands and Counts Output to GT
    // ConcElec FPGA                                                            
    m_blockUnpackFn[0x683] = &GctFormatTranslateMCLegacy::blockToGctEmCandsAndEnergySums;    // ConcElec: EM Cands and Energy Sums Output to GT
    // Electron Leaf FPGAs                                                      
    m_blockUnpackFn[0x804] = &GctFormatTranslateMCLegacy::blockToFibresAndToRctEmCand;       // Leaf0ElecPosEtaU1: Raw Input
    m_blockUnpackFn[0x884] = &GctFormatTranslateMCLegacy::blockToFibresAndToRctEmCand;       // Leaf0ElecPosEtaU2: Raw Input
    m_blockUnpackFn[0xc04] = &GctFormatTranslateMCLegacy::blockToFibresAndToRctEmCand;       // Leaf0ElecNegEtaU1: Raw Input
    m_blockUnpackFn[0xc84] = &GctFormatTranslateMCLegacy::blockToFibresAndToRctEmCand;       // Leaf0ElecNegEtaU2: Raw Input


    /*** Setup RCT Em Crate Map ***/
    m_rctEmCrate[0x804] = 13;
    m_rctEmCrate[0x884] = 9;
    m_rctEmCrate[0xc04] = 4;
    m_rctEmCrate[0xc84] = 0;


    /*** Setup RCT jet crate map. ***/
    // No entries required!


    /*** Setup Block ID map for pipeline payload positions of isolated Internal EM Cands. ***/
    // No entries required!
  }
}

GctFormatTranslateMCLegacy::~GctFormatTranslateMCLegacy()
{
}

GctBlockHeader GctFormatTranslateMCLegacy::generateBlockHeader(const unsigned char * data) const
{
  // Turn the four 8-bit header words into the full 32-bit header.
  uint32_t hdr = data[0] + (data[1]<<8) + (data[2]<<16) + (data[3]<<24);

  //  Bit mapping of header:
  //  ----------------------
  //  11:0   => block_id  Unique pipeline identifier.
  //   - 3:0    =>> pipe_id There can be up to 16 different pipelines per FPGA.
  //   - 6:4    =>> reserved  Do not use yet. Set to zero.
  //   - 11:7   =>> fpga geograpical add  The VME geographical address of the FPGA.
  //  15:12  => event_id  Determined locally.  Not reset by Resync.
  //  19:16  => number_of_time_samples  If time samples 15 or more then value = 15.
  //  31:20  => event_bxId  The bunch crossing the data was recorded.

  uint32_t blockId = hdr & 0xfff;
  uint32_t blockLength = 0;  // Set to zero until we know it's a valid block
  uint32_t nSamples = (hdr>>16) & 0xf;
  uint32_t bxId = (hdr>>20) & 0xfff;
  uint32_t eventId = (hdr>>12) & 0xf;
  bool valid = (blockLengthMap().find(blockId) != blockLengthMap().end());

  if(valid) { blockLength = blockLengthMap().find(blockId)->second; }
  
  return GctBlockHeader(blockId, blockLength, nSamples, bxId, eventId, valid);  
}

// conversion
bool GctFormatTranslateMCLegacy::convertBlock(const unsigned char * data, const GctBlockHeader& hdr)
{
  // if the block has no time samples, don't bother with it.
  if ( hdr.nSamples() < 1 ) { return true; }

  if(!checkBlock(hdr)) { return false; }  // Check the block to see if it's possible to unpack.

  // The header validity check above will protect against
  // the map::find() method returning the end of the map,
  // assuming the block header definitions are up-to-date.
  (this->*m_blockUnpackFn.find(hdr.blockId())->second)(data, hdr);  // Calls the correct unpack function, based on block ID.
  
  return true;
}

// Output EM Candidates and energy sums packing
void GctFormatTranslateMCLegacy::writeGctOutEmAndEnergyBlock(unsigned char * d,
                                                             const L1GctEmCandCollection* iso,
                                                             const L1GctEmCandCollection* nonIso,
                                                             const L1GctEtTotalCollection* etTotal,
                                                             const L1GctEtHadCollection* etHad,
                                                             const L1GctEtMissCollection* etMiss)
{
  // Set up a vector of the collections for easy iteration.
  vector<const L1GctEmCandCollection*> emCands(NUM_EM_CAND_CATEGORIES);
  emCands.at(NON_ISO_EM_CANDS)=nonIso;
  emCands.at(ISO_EM_CANDS)=iso;
  
  /* To hold the offsets within the EM candidate collections for the bx=0 candidates.
   * The capture index doesn't seem to get set properly by the emulator, so take the
   * first bx=0 cand as the highest energy EM cand, and the fourth as the lowest. */
  vector<unsigned> bx0EmCandOffsets(NUM_EM_CAND_CATEGORIES);

  // Loop over the different catagories of EM cands to find the bx=0 offsets.
  for(unsigned int iCat = 0 ; iCat < NUM_EM_CAND_CATEGORIES ; ++iCat)
  {
    const L1GctEmCandCollection * cands = emCands.at(iCat);
    unsigned& offset = bx0EmCandOffsets.at(iCat);
    if(!findBx0OffsetInCollection(offset, cands)) { LogDebug("GCT") << "No EM candidates with bx=0!\nAborting packing of GCT EM Cand and Energy Sum Output!"; return; }
    if((cands->size()-offset) < 4) { LogDebug("GCT") << "Insufficient EM candidates with bx=0!\nAborting packing of GCT EM Cand and Energy Sum Output!"; return; }
  }
  
  unsigned bx0EtTotalOffset, bx0EtHadOffset, bx0EtMissOffset;
  if(!findBx0OffsetInCollection(bx0EtTotalOffset, etTotal)) { LogDebug("GCT") << "No Et Total value for bx=0!\nAborting packing of GCT EM Cand and Energy Sum Output!"; return; }
  if(!findBx0OffsetInCollection(bx0EtHadOffset, etHad)) { LogDebug("GCT") << "No Et Hadronic value for bx=0!\nAborting packing of GCT EM Cand and Energy Sum Output!"; return; }
  if(!findBx0OffsetInCollection(bx0EtMissOffset, etMiss)) { LogDebug("GCT") << "No Et Miss value for bx=0!\nAborting packing of GCT EM Cand and Energy Sum Output!"; return; } 
  
  // We should now have all requisite data, so we can get on with packing

  unsigned nSamples = 1; // ** NOTE can only currenly do 1 timesample! **
  
  // write header
  writeRawHeader(d, 0x683, nSamples);   
  
  d=d+4;  // move to the block payload.

  // FIRST DO EM CANDS

  // re-interpret payload pointer to 16 bit.
  uint16_t * p16 = reinterpret_cast<uint16_t *>(d);

  for (unsigned iCat=0; iCat < NUM_EM_CAND_CATEGORIES; ++iCat)   // loop over non-iso/iso candidates categories
  {
    const L1GctEmCandCollection * em = emCands.at(iCat);   // The current category of EM cands.
    const unsigned bx0Offset = bx0EmCandOffsets.at(iCat);  // The offset in the EM cand collection to the bx=0 cands.
    
    uint16_t * cand = p16 + (iCat*4);

    *cand = em->at(bx0Offset).raw();
    cand++;
    *cand = em->at(bx0Offset + 2).raw();
    cand += nSamples;
    *cand = em->at(bx0Offset + 1).raw();
    cand++;
    *cand = em->at(bx0Offset + 3).raw();
  }
  
  // NOW DO ENERGY SUMS
  // assumes these are all 1-object collections, ie. central BX only
  p16+=8;  // Move past EM cands
  *p16 = etTotal->at(bx0EtTotalOffset).raw();  // Et Total - 16 bits.
  p16++;
  *p16 = etHad->at(bx0EtHadOffset).raw();  // Et Hadronic - next 16 bits
  p16++;
  uint32_t * p32 = reinterpret_cast<uint32_t *>(p16);  // For writing Missing Et (32-bit raw data)
  *p32 = etMiss->at(bx0EtMissOffset).raw();  // Et Miss on final 32 bits of block payload.
}

void GctFormatTranslateMCLegacy::writeGctOutJetBlock(unsigned char * d,
                                                     const L1GctJetCandCollection* cenJets,
                                                     const L1GctJetCandCollection* forJets,
                                                     const L1GctJetCandCollection* tauJets, 
                                                     const L1GctHFRingEtSumsCollection* hfRingSums,
                                                     const L1GctHFBitCountsCollection* hfBitCounts,
                                                     const L1GctHtMissCollection* htMiss)
{
  // Set up a vector of the collections for easy iteration.
  vector<const L1GctJetCandCollection*> jets(NUM_JET_CATEGORIES);
  jets.at(CENTRAL_JETS)=cenJets;
  jets.at(FORWARD_JETS)=forJets;
  jets.at(TAU_JETS)=tauJets;

  /* To hold the offsets within the three jet cand collections for the bx=0 jets.
   * The capture index doesn't seem to get set properly by the emulator, so take the
   * first bx=0 jet as the highest energy jet, and the fourth as the lowest. */
  vector<unsigned> bx0JetCandOffsets(NUM_JET_CATEGORIES);

  // Loop over the different catagories of jets to find the bx=0 offsets.
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATEGORIES ; ++iCat)
  {
    const L1GctJetCandCollection * jetCands = jets.at(iCat);
    unsigned& offset = bx0JetCandOffsets.at(iCat);
    if(!findBx0OffsetInCollection(offset, jetCands)) { LogDebug("GCT") << "No jet candidates with bx=0!\nAborting packing of GCT Jet Output!"; return; }
    if((jetCands->size()-offset) < 4) { LogDebug("GCT") << "Insufficient jet candidates with bx=0!\nAborting packing of GCT Jet Output!"; return; }
  }
  
  // Now find the collection offsets for the HfRingSums, HfBitCounts, and HtMiss with bx=0
  unsigned bx0HfRingSumsOffset, bx0HfBitCountsOffset, bx0HtMissOffset;
  if(!findBx0OffsetInCollection(bx0HfRingSumsOffset, hfRingSums)) { LogDebug("GCT") << "No ring sums with bx=0!\nAborting packing of GCT Jet Output!"; return; }
  if(!findBx0OffsetInCollection(bx0HfBitCountsOffset, hfBitCounts)) { LogDebug("GCT") << "No bit counts with bx=0!\nAborting packing of GCT Jet Output!"; return; }
  if(!findBx0OffsetInCollection(bx0HtMissOffset, htMiss)) { LogDebug("GCT") << "No missing Ht with bx=0!\nAborting packing of GCT Jet Output!"; return; }

  // Now write the header, as we should now have all requisite data.
  writeRawHeader(d, 0x583, 1);  // ** NOTE can only currenly do 1 timesample! **
  
  d=d+4;  // move forward past the block header to the block payload.

  // FIRST DO JET CANDS
  // re-interpret pointer to 16 bits - the space allocated for each Jet candidate.
  uint16_t * p16 = reinterpret_cast<uint16_t *>(d);
  
  const unsigned categoryOffset = 4;  // Offset to jump from one jet category to the next.
  const unsigned nextCandPairOffset = 2;  // Offset to jump to next candidate pair.

  // Loop over the different catagories of jets
  for(unsigned iCat = 0 ; iCat < NUM_JET_CATEGORIES ; ++iCat)
  {
    const L1GctJetCandCollection * jetCands = jets.at(iCat); // The current category of jet cands.
    const unsigned cand0Offset = iCat*categoryOffset;       // the offset on p16 to get the rank 0 Jet Cand of the correct category.
    const unsigned bx0Offset = bx0JetCandOffsets.at(iCat);  // The offset in the jet cand collection to the bx=0 jets.
    
    p16[cand0Offset] = jetCands->at(bx0Offset).raw();  // rank 0 jet in bx=0
    p16[cand0Offset + nextCandPairOffset] = jetCands->at(bx0Offset + 1).raw(); // rank 1 jet in bx=0
    p16[cand0Offset + 1] = jetCands->at(bx0Offset + 2).raw(); // rank 2 jet in bx=0
    p16[cand0Offset + nextCandPairOffset + 1] = jetCands->at(bx0Offset + 3).raw(); // rank 3 jet in bx=0.
  }
  
  // NOW DO JET COUNTS
  d=d+24;  // move forward past the jet cands to the jet counts section

  // re-interpret pointer to 32 bit.
  uint32_t * p32 = reinterpret_cast<uint32_t *>(d);
  
  uint32_t tmp = hfBitCounts->at(bx0HfBitCountsOffset).raw() & 0xfff;
  tmp |= hfRingSums->at(bx0HfRingSumsOffset).etSum(0)<<12;
  tmp |= hfRingSums->at(bx0HfRingSumsOffset).etSum(1)<<16;
  tmp |= hfRingSums->at(bx0HfRingSumsOffset).etSum(2)<<19;
  tmp |= hfRingSums->at(bx0HfRingSumsOffset).etSum(3)<<22;
  p32[0] = tmp;
  
  const L1GctHtMiss& bx0HtMiss = htMiss->at(bx0HtMissOffset);
  uint32_t htMissRaw = 0x5555c000 |
                       (bx0HtMiss.overFlow() ? 0x1000 : 0x0000) |
                       ((bx0HtMiss.et() & 0x7f) << 5) |
                       ((bx0HtMiss.phi() & 0x1f));
  
  p32[1] = htMissRaw;
}

void GctFormatTranslateMCLegacy::writeRctEmCandBlocks(unsigned char * d, const L1CaloEmCollection * rctEm)
{
  // This method is one giant "temporary" hack for CMSSW_1_8_X and CMSSW_2_0_0.

  if(rctEm->size() == 0 || rctEm->size()%144 != 0)  // Should be 18 crates * 2 types (iso/noniso) * 4 electrons = 144 for 1 bx.
  {
    LogDebug("GCT") << "Block pack error: bad L1CaloEmCollection size detected!\n"
                    << "Aborting packing of RCT EM Cand data!";
    return;
  }

  // Need 18 sets of EM fibre data, since 18 RCT crates  
  SourceCardRouting::EmuToSfpData emuToSfpData[18];

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
    srcCardRouting().EMUtoSFP(emuToSfpData[c].eIsoRank, emuToSfpData[c].eIsoCardId, emuToSfpData[c].eIsoRegionId,
                              emuToSfpData[c].eNonIsoRank, emuToSfpData[c].eNonIsoCardId, emuToSfpData[c].eNonIsoRegionId,
                              emuToSfpData[c].mipBits, emuToSfpData[c].qBits, emuToSfpData[c].sfp);
  }
  
  // Now pack up the data into the RAW format.
  BlkToRctCrateMap::iterator blockStartCrateIter;
  for(blockStartCrateIter = rctEmCrateMap().begin() ; blockStartCrateIter != rctEmCrateMap().end() ; ++blockStartCrateIter)
  {
    unsigned blockId = blockStartCrateIter->first;
    unsigned startCrate = blockStartCrateIter->second;
    unsigned blockLength_32bit = blockLengthMap()[blockId];
    
    writeRawHeader(d, blockId, 1);
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

void GctFormatTranslateMCLegacy::writeAllRctCaloRegionBlock(unsigned char * d, const L1CaloRegionCollection * rctCalo)
{
  // This method is one giant "temporary" hack for CMSSW_1_8_X and CMSSW_2_0_0.

  if(rctCalo->size() == 0 || rctCalo->size()%396 != 0)  // Should be 396 calo regions for 1 bx.
  {
    LogDebug("GCT") << "Block pack error: bad L1CaloRegionCollection size detected!\n"
                    << "Aborting packing of RCT Calo Region data!";
    return;
  }

  writeRawHeader(d, 0x0ff, 1);
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


// PROTECTED METHODS

uint32_t GctFormatTranslateMCLegacy::generateRawHeader(const uint32_t blockId,
                                                       const uint32_t nSamples,
                                                       const uint32_t bxId,
                                                       const uint32_t eventId) const
{
  //  Bit mapping of header:
  //  ----------------------
  //  11:0   => block_id  Unique pipeline identifier.
  //   - 3:0    =>> pipe_id There can be up to 16 different pipelines per FPGA.
  //   - 6:4    =>> reserved  Do not use yet. Set to zero.
  //   - 11:7   =>> fpga geograpical add  The VME geographical address of the FPGA.
  //  15:12  => event_id  Determined locally.  Not reset by Resync.
  //  19:16  => number_of_time_samples  If time samples 15 or more then value = 15.
  //  31:20  => event_bxId  The bunch crossing the data was recorded.

  return ((bxId & 0xfff) << 20) | ((nSamples & 0xf) << 16) | ((eventId & 0xf) << 12) | (blockId & 0xfff);
}


// PRIVATE METHODS

// Output EM Candidates unpacking
void GctFormatTranslateMCLegacy::blockToGctEmCandsAndEnergySums(const unsigned char * d, const GctBlockHeader& hdr)
{
  const unsigned int id = hdr.blockId();
  const unsigned int nSamples = hdr.nSamples();

  // Re-interpret pointer.  p16 will be pointing at the 16 bit word that
  // contains the rank0 non-isolated electron of the zeroth time-sample.
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);

  // UNPACK EM CANDS

  const unsigned int emCandCategoryOffset = nSamples * 4;  // Offset to jump from the non-iso electrons to the isolated ones.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  unsigned int samplesToUnpack = 1;
  if(!hltMode()) { samplesToUnpack = nSamples; }  // Only if not running in HLT mode do we want more than 1 timesample. 

  for (unsigned int iso=0; iso<2; ++iso)  // loop over non-iso/iso candidate pairs
  {
    bool isoFlag = (iso==1);

    // Get the correct collection to put them in.
    L1GctEmCandCollection* em;
    if (isoFlag) { em = colls()->gctIsoEm(); }
    else { em = colls()->gctNonIsoEm(); }

    for (unsigned int bx=0; bx<samplesToUnpack; ++bx) // loop over time samples
    {
      // cand0Offset will give the offset on p16 to get the rank 0 candidate
      // of the correct category and timesample.
      const unsigned int cand0Offset = iso*emCandCategoryOffset + bx*2;

      em->push_back(L1GctEmCand(p16[cand0Offset], isoFlag, id, 0, bx));  // rank0 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + timeSampleOffset], isoFlag, id, 1, bx));  // rank1 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + 1], isoFlag, id, 2, bx));  // rank2 electron
      em->push_back(L1GctEmCand(p16[cand0Offset + timeSampleOffset + 1], isoFlag, id, 3, bx));  // rank3 electron
    }
  }

  p16 += emCandCategoryOffset * 2;  // Move the pointer over the data we've already unpacked.

  // UNPACK ENERGY SUMS
  // NOTE: we are only unpacking one timesample of these currently!

  colls()->gctEtTot()->push_back(L1GctEtTotal(p16[0]));  // Et total (timesample 0).
  colls()->gctEtHad()->push_back(L1GctEtHad(p16[1]));  // Et hadronic (timesample 0).

  // 32-bit pointer for getting Missing Et.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  colls()->gctEtMiss()->push_back(L1GctEtMiss(p32[nSamples])); // Et Miss (timesample 0).
}

void GctFormatTranslateMCLegacy::blockToGctJetCandsAndCounts(const unsigned char * d, const GctBlockHeader& hdr)
{
  const unsigned int id = hdr.blockId();  // Capture block ID.
  const unsigned int nSamples = hdr.nSamples();  // Number of time-samples.

  // Re-interpret block payload pointer to 16 bits so it sees one candidate at a time.
  // p16 points to the start of the block payload, at the rank0 tau jet candidate.
  const uint16_t * p16 = reinterpret_cast<const uint16_t *>(d);

  // UNPACK JET CANDS

  const unsigned int jetCandCategoryOffset = nSamples * 4;  // Offset to jump from one jet category to the next.
  const unsigned int timeSampleOffset = nSamples * 2;  // Offset to jump to next candidate pair in the same time-sample.

  unsigned int samplesToUnpack = 1;
  if(!hltMode()) { samplesToUnpack = nSamples; }  // Only if not running in HLT mode do we want more than 1 timesample. 

  // Loop over the different catagories of jets
  for(unsigned int iCat = 0 ; iCat < NUM_JET_CATEGORIES ; ++iCat)
  {
    L1GctJetCandCollection * const jets = gctJets(iCat);
    assert(jets->empty()); // The supplied vector should be empty.

    bool tauflag = (iCat == TAU_JETS);
    bool forwardFlag = (iCat == FORWARD_JETS);

    // Loop over the different timesamples (bunch crossings).
    for(unsigned int bx = 0 ; bx < samplesToUnpack ; ++bx)
    {
      // cand0Offset will give the offset on p16 to get the rank 0 Jet Cand of the correct category and timesample.
      const unsigned int cand0Offset = iCat*jetCandCategoryOffset + bx*2;

      // Rank 0 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset], tauflag, forwardFlag, id, 0, bx));
      // Rank 1 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset], tauflag, forwardFlag, id, 1, bx));
      // Rank 2 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset + 1],  tauflag, forwardFlag, id, 2, bx));
      // Rank 3 Jet.
      jets->push_back(L1GctJetCand(p16[cand0Offset + timeSampleOffset + 1], tauflag, forwardFlag, id, 3, bx));
    }
  }

  p16 += NUM_JET_CATEGORIES * jetCandCategoryOffset; // Move the pointer over the data we've already unpacked.

  // NOW UNPACK: HFBitCounts, HFRingEtSums and Missing Ht
  // NOTE: we are only unpacking one timesample of these currently!

  // Re-interpret block payload pointer to 32 bits so it sees six jet counts at a time.
  const uint32_t * p32 = reinterpret_cast<const uint32_t *>(p16);

  // Channel 0 carries both HF counts and sums
  colls()->gctHfBitCounts()->push_back(L1GctHFBitCounts::fromConcHFBitCounts(id,6,0,p32[0])); 
  colls()->gctHfRingEtSums()->push_back(L1GctHFRingEtSums::fromConcRingSums(id,6,0,p32[0]));

  // Channel 1 carries Missing HT.
  colls()->gctHtMiss()->push_back(L1GctHtMiss(p32[nSamples], 0));
}

// Input EM Candidates unpacking
// this is the last time I deal the RCT bit assignment travesty!!!
void GctFormatTranslateMCLegacy::blockToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of RCT EM Cands"; return; }

  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

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
  for (unsigned int crate=rctEmCrateMap()[id]; crate<rctEmCrateMap()[id]+length/3; ++crate) {

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
    srcCardRouting().SFPtoEMU(eIsoRank, eIsoCard, eIsoRgn, eNonIsoRank, eNonIsoCard, eNonIsoRgn, MIPbits, QBits, sfp);
    
    // create EM cands
    for (unsigned short int i=0; i<4; ++i) {
      colls()->rctEm()->push_back( L1CaloEmCand( eIsoRank[i], eIsoRgn[i], eIsoCard[i], crate, true, i, bx) );
    }
    for (unsigned short int i=0; i<4; ++i) {
      colls()->rctEm()->push_back( L1CaloEmCand( eNonIsoRank[i], eNonIsoRgn[i], eNonIsoCard[i], crate, false, i, bx) );
    }
  }
}

// Fibre unpacking
void GctFormatTranslateMCLegacy::blockToFibres(const unsigned char * d, const GctBlockHeader& hdr)
{
  // Don't want to do this in HLT optimisation mode!
  if(hltMode()) { LogDebug("GCT") << "HLT mode - skipping unpack of GCT Fibres"; return; }
  
  unsigned int id = hdr.blockId();
  unsigned int nSamples = hdr.nSamples();
  unsigned int length = hdr.blockLength();

  // re-interpret pointer
  uint32_t * p = reinterpret_cast<uint32_t *>(const_cast<unsigned char *>(d));

  for (unsigned int i=0; i<length; ++i) {
    for (unsigned int bx=0; bx<nSamples; ++bx) {
      colls()->gctFibres()->push_back( L1GctFibreWord(*p, id, i, bx) );
      ++p;
    }
  } 
}

void GctFormatTranslateMCLegacy::blockToFibresAndToRctEmCand(const unsigned char * d, const GctBlockHeader& hdr)
{
  this->blockToRctEmCand(d, hdr);
  this->blockToFibres(d, hdr);
}

void GctFormatTranslateMCLegacy::blockToAllRctCaloRegions(const unsigned char * d, const GctBlockHeader& hdr)
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
          colls()->rctCalo()->push_back(L1CaloRegion(*p16, id.ieta(), id.iphi(), iSample));
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
          colls()->rctCalo()->push_back(L1CaloRegion(*p16, id.ieta(), id.iphi(), iSample));
          ++p16; //advance pointer
        }
      }
    }
  }
}

template <typename Collection> 
bool GctFormatTranslateMCLegacy::findBx0OffsetInCollection(unsigned& bx0Offset, const Collection* coll)
{
  bool foundBx0 = false;
  unsigned size = coll->size();
  for(bx0Offset = 0 ; bx0Offset < size ; ++bx0Offset)
  {
    if(coll->at(bx0Offset).bx() == 0) { foundBx0 = true; break; }
  }
  return foundBx0;
}
