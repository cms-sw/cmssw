#include "EventFilter/GctRawToDigi/src/GctFormatTranslateBase.h"

// Framework headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// INITIALISE STATICS
const std::string GctFormatTranslateBase::INVALID_BLOCK_HEADER_STR = "UNKNOWN/INVALID BLOCK HEADER";


// PUBLIC METHODS

GctFormatTranslateBase::GctFormatTranslateBase(bool hltMode, bool unpackSharedRegions):
  m_collections(nullptr),
  m_hltMode(hltMode),
  m_unpackSharedRegions(unpackSharedRegions),
  m_srcCardRouting(),
  m_packingBxId(0),
  m_packingEventId(0)
{
}

GctFormatTranslateBase::~GctFormatTranslateBase() { }

const std::string& GctFormatTranslateBase::getBlockDescription(const GctBlockHeader& header) const
{
  if(!header.valid()) { return INVALID_BLOCK_HEADER_STR; }
  return blockNameMap().find(header.blockId())->second;
}


// PROTECTED METHODS

L1GctJetCandCollection * const GctFormatTranslateBase::gctJets(const unsigned cat) const
{
  switch(cat)
  {
    case TAU_JETS: return colls()->gctTauJets();
    case FORWARD_JETS: return colls()->gctForJets();
    default: return colls()->gctCenJets();
  } 
}

void GctFormatTranslateBase::writeRawHeader(unsigned char * data, uint32_t blockId, uint32_t nSamples) const
{
  uint32_t hdr = generateRawHeader(blockId, nSamples, packingBxId(), packingEventId());
  uint32_t * p = reinterpret_cast<uint32_t*>(const_cast<unsigned char *>(data));
  *p = hdr;
}

bool GctFormatTranslateBase::checkBlock(const GctBlockHeader& hdr) const
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


