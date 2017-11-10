#include <cstdint>
#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"

using namespace gem;

uint64_t AMC13Event::getCDFHeader()
{
  return (static_cast<uint64_t>(m_cb5 & 0x0f) << 60) |
    (static_cast<uint64_t>(m_Evt_ty & 0x0f) << 56) |
    (static_cast<uint64_t>(m_LV1_id & 0x00ffffff) << 32) |
    (static_cast<uint64_t>(m_BX_id & 0x0fff) << 20) |
    (static_cast<uint64_t>(m_Source_id & 0x0fff) << 8);
}

void AMC13Event::setCDFHeader(uint8_t cb5, uint8_t Evt_ty, uint32_t LV1_id, uint16_t BX_id, uint16_t Source_id)
{
  m_cb5 = cb5;
  m_Evt_ty = Evt_ty;
  m_LV1_id = LV1_id;
  m_BX_id = BX_id;
  m_Source_id = Source_id;
}

void AMC13Event::setCDFHeader(uint64_t word)
{
  m_cb5 = 0x0f & (word >> 60);
  m_Evt_ty = 0x0f & (word >> 56);
  m_LV1_id = 0x00ffffff & (word >> 32);
  m_BX_id = 0x0fff & (word >> 20);
  m_Source_id = 0x0fff & (word >> 8);
}

uint64_t AMC13Event::getAMC13header()
{
  return (static_cast<uint64_t>(m_CalTyp & 0x0f) << 56) |
    (static_cast<uint64_t>(m_nAMC & 0x0f) << 52) |
    (static_cast<uint64_t>(m_OrN) << 4) |
    (static_cast<uint64_t>(m_cb0 & 0x0f));
}

void AMC13Event::setAMC13header(uint8_t CalTyp, uint8_t nAMC, uint32_t OrN, uint8_t cb0)
{
  m_CalTyp = CalTyp;
  m_nAMC = nAMC;
  m_OrN = OrN;
  m_cb0 = cb0;
}

void AMC13Event::setAMC13header(uint64_t word)
{
  m_CalTyp = 0x0f & (word >> 56);
  m_nAMC = 0x0f & (word >> 52);
  m_OrN = word >> 4;
  m_cb0 = 0x0f & word;
}

std::vector<uint64_t> AMC13Event::getAMCheader()
{
  std::vector<uint64_t> amcHeaders;
  for (unsigned int i = 0; i < m_AMC_size.size(); ++i){
    uint64_t word =
      (static_cast<uint64_t>(m_AMC_size[i] & 0x00ffffff) << 32) |
      (static_cast<uint64_t>(m_Blk_No[i] & 0xff) << 20) |
      (static_cast<uint64_t>(m_AMC_No[i] & 0x0f) << 16) |
      (static_cast<uint64_t>(m_BoardID[i] & 0xffff));
    amcHeaders.push_back(word);
  }
  return amcHeaders;
}

void AMC13Event::addAMCheader(uint32_t AMC_size, uint8_t Blk_No, uint8_t AMC_No, uint16_t BoardID)
{
  m_AMC_size.push_back(AMC_size);
  m_Blk_No.push_back(Blk_No);
  m_AMC_No.push_back(AMC_No);
  m_BoardID.push_back(BoardID);
}

void AMC13Event::addAMCheader(uint64_t word)
{
  m_AMC_size.push_back(0x00ffffff&(word>>32));
  m_Blk_No.push_back(0xff&(word>>20));
  m_AMC_No.push_back(0x0f&(word>>16));
  m_BoardID.push_back(0xffff&word);
}

uint64_t AMC13Event::getAMC13trailer()
{
  return (static_cast<uint64_t>(m_CRC_amc13) << 32) |
    (static_cast<uint64_t>(m_Blk_NoT & 0xff) << 20) |
    (static_cast<uint64_t>(m_LV1_idT & 0xff) << 12) |
    (static_cast<uint64_t>(m_BX_idT & 0x0fff));
}

void AMC13Event::setAMC13trailer(uint32_t CRC_amc13, uint8_t Blk_NoT, uint8_t LV1_idT, uint16_t BX_idT)
{
  m_CRC_amc13 = CRC_amc13;
  m_Blk_NoT = Blk_NoT;
  m_LV1_idT = LV1_idT;
  m_BX_idT = BX_idT;
}

void AMC13Event::setAMC13trailer(uint64_t word)
{
  m_CRC_amc13 = word >> 32;
  m_Blk_NoT = 0xff & (word >> 20);
  m_LV1_idT = 0xff & (word >> 12);
  m_BX_idT = 0x0fff & word;
}

uint64_t AMC13Event::getCDFTrailer()
{
  return (static_cast<uint64_t>(m_cbA & 0x0f) << 60) |
    (static_cast<uint64_t>(m_EvtLength & 0x00ffffff) << 32) |
    (static_cast<uint64_t>(m_CRC_cdf & 0xffff) << 16);
}

void AMC13Event::setCDFTrailer(uint8_t cbA, uint32_t EvtLength, uint16_t CRC_cdf)
{
  m_cbA = cbA;
  m_EvtLength = EvtLength;
  m_CRC_cdf = CRC_cdf;
}

void AMC13Event::setCDFTrailer(uint64_t word)
{
  m_cbA = 0x0f & (word >> 60);
  m_EvtLength = 0x00ffffff & (word >> 32);
  m_CRC_cdf = 0xffff & (word >> 16);
}
