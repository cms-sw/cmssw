#include "EventFilter/HGCalRawToDigi/interface/HGCalRawDataPackingTools.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalRawDataDefinitions.h"

std::vector<uint32_t> hgcal::econd::addChannelData(uint8_t& msb,
                                                   uint16_t tctp,
                                                   uint16_t adc,
                                                   uint16_t tot,
                                                   uint16_t adcm,
                                                   uint16_t toa,
                                                   bool passZS,
                                                   bool passZSm1,
                                                   bool hasToA,
                                                   bool charmode) {
  std::vector<uint32_t> newWords(0);

  if (msb == 0)
    msb = 32;

  //characterization mode is easy
  if (charmode) {
    newWords.push_back((tctp & 0x3) << 30 | (adc & 0x3ff) << 20 | (tot & 0x3ff) << 10 | (toa & 0x3ff));

    //normal mode will depend on the TcTp and ZS states
  } else {
    uint32_t rawWord(0);
    uint8_t nbits(0);

    switch (tctp) {
      case 0x0:
        if (passZS && passZSm1 && !hasToA) {
          rawWord = ((adcm & 0x3ff) << 10) | (adc & 0x3ff);
          nbits = 24;
        } else if (passZS && !passZSm1 && !hasToA) {
          rawWord = (1 << 10) | (adc & 0x3ff);
          nbits = 16;
        } else if (passZS && !passZSm1 && hasToA) {
          rawWord = (3 << 20) | ((adc & 0x3ff) << 10) | (toa & 0x3ff);
          nbits = 24;
        } else if (passZS && passZSm1 && hasToA) {
          rawWord = (1 << 30) | ((adcm & 0x3ff) << 20) | ((adc & 0x3ff) << 10) | (toa & 0x3ff);
          nbits = 32;
        }
        break;
      case 0x1:
        rawWord = (2 << 20) | ((adcm & 0x3ff) << 10) | (adc & 0x3ff);
        nbits = 24;
        break;
      case 0x2:
        rawWord = (2 << 30) | ((adcm & 0x3ff) << 20) | ((adc & 0x3ff) << 10) | (toa & 0x3ff);
        nbits = 32;
        break;
      case 0x3:
        rawWord = (3 << 30) | ((adcm & 0x3ff) << 20) | ((tot & 0x3ff) << 10) | (toa & 0x3ff);
        nbits = 32;
        break;
    }

    //pack raw word to 32b starting at the required msb
    //if it still fits, it sits in the current word
    if (msb >= nbits) {
      uint32_t mask = 0xffffffff >> (32 - nbits);
      if (nbits > 0) {
        newWords.push_back((rawWord & mask) << (msb - nbits));
        msb = msb == nbits ? 32 : msb - nbits;
      }
      //if it doesn't fit start a new word
      //    32                         0 32                            0 32
      //    |           .    msb        |  nbits-msb  . 32-(nbits-msb)  |
    } else {
      uint8_t spill(nbits - msb);
      uint32_t rawWord1(rawWord >> spill);
      newWords.push_back(rawWord1);

      uint32_t rawWord2(rawWord << (32 - spill));
      newWords.push_back(rawWord2);

      msb = spill != 32 ? 32 - spill : 32;
    }
  }

  return newWords;
}

//
std::vector<uint32_t> hgcal::econd::eRxSubPacketHeader(
    uint16_t stat, uint16_t ham, bool bitE, uint16_t cm0, uint16_t cm1, std::vector<bool> chmap) {

  uint64_t chmap64b(0);
  for(size_t i=0; i<65; i++) chmap64b |= (chmap[i]<<i);
  return hgcal::econd::eRxSubPacketHeader(stat,ham,bitE,cm0,cm1,chmap64b);

}


//
std::vector<uint32_t> hgcal::econd::eRxSubPacketHeader(
    uint16_t stat, uint16_t ham, bool bitE, uint16_t cm0, uint16_t cm1, uint64_t chmap) {

  uint32_t header((stat & hgcal::ECOND_FRAME::ERXSTAT_MASK) << hgcal::ECOND_FRAME::ERXSTAT_POS |
                  (ham & hgcal::ECOND_FRAME::ERXHAM_MASK) << hgcal::ECOND_FRAME::ERXHAM_POS |
                  (cm0 & hgcal::ECOND_FRAME::COMMONMODE0_MASK) << hgcal::ECOND_FRAME::COMMONMODE0_POS |
                  (cm1 & hgcal::ECOND_FRAME::COMMONMODE1_MASK) << hgcal::ECOND_FRAME::COMMONMODE1_POS);

  std::vector<uint32_t> newWords(1, header);

  //summarize the channel status map
  uint32_t chmapw0(chmap & hgcal::ECOND_FRAME::CHMAP0_MASK);
  uint32_t chmapw1( (chmap >> 32) & hgcal::ECOND_FRAME::CHMAP32_MASK);

  //add the channel map
  if (chmapw0 == 0 && chmapw1 == 0) {
    newWords[0] |= (bitE << hgcal::ECOND_FRAME::ERX_E_POS);
    newWords[0] |= (1 << hgcal::ECOND_FRAME::ERXFORMAT_POS);  //raise the F bit (empty eRX)
  } else {
    newWords[0] |= (chmapw0 & hgcal::ECOND_FRAME::CHMAP32_MASK) << hgcal::ECOND_FRAME::CHMAP32_POS;
    newWords.push_back((chmapw1 & hgcal::ECOND_FRAME::CHMAP0_MASK) << hgcal::ECOND_FRAME::CHMAP0_POS);
  }

  return newWords;


}


//
std::vector<uint32_t> hgcal::econd::eventPacketHeader(uint16_t header,
                                                      uint16_t payload,
                                                      bool bitP,
                                                      bool bitE,
                                                      uint8_t ht,
                                                      uint8_t ebo,
                                                      bool bitM,
                                                      bool bitT,
                                                      uint8_t ehHam,
                                                      uint16_t bx,
                                                      uint16_t l1a,
                                                      uint8_t orb,
                                                      bool bitS,
                                                      uint8_t rr,
                                                      uint8_t ehCRC) {
  std::vector<uint32_t> words(2, 0);

  words[0] = (header & hgcal::ECOND_FRAME::HEADER_MASK) << hgcal::ECOND_FRAME::HEADER_POS |
             (payload & hgcal::ECOND_FRAME::PAYLOAD_MASK) << hgcal::ECOND_FRAME::PAYLOAD_POS |
             bitP << hgcal::ECOND_FRAME::BITP_POS | bitE << hgcal::ECOND_FRAME::BITE_POS |
             (ht & hgcal::ECOND_FRAME::HT_MASK) << hgcal::ECOND_FRAME::HT_POS |
             (ebo & hgcal::ECOND_FRAME::EBO_MASK) << hgcal::ECOND_FRAME::EBO_POS |
             bitM << hgcal::ECOND_FRAME::BITM_POS | bitT << hgcal::ECOND_FRAME::BITT_POS |
             (ehHam & hgcal::ECOND_FRAME::EHHAM_MASK) << hgcal::ECOND_FRAME::EHHAM_POS;

  words[1] = (bx & hgcal::ECOND_FRAME::BX_MASK) << hgcal::ECOND_FRAME::BX_POS |
             (l1a & hgcal::ECOND_FRAME::L1A_MASK) << hgcal::ECOND_FRAME::L1A_POS |
             (orb & hgcal::ECOND_FRAME::ORBIT_MASK) << hgcal::ECOND_FRAME::ORBIT_POS |
             bitS << hgcal::ECOND_FRAME::BITS_POS | (rr & hgcal::ECOND_FRAME::RR_MASK) << hgcal::ECOND_FRAME::RR_POS |
             (ehCRC & hgcal::ECOND_FRAME::EHCRC_MASK) << hgcal::ECOND_FRAME::EHCRC_POS;

  return words;
}

uint32_t hgcal::econd::buildIdleWord(uint8_t bufStat, uint8_t err, uint8_t rr, uint32_t progPattern) {
  return (progPattern & 0xffffff) << 8 | (rr & 0x3) << 6 | (err & 0x7) << 3 | (bufStat & 0x7) << 0;
}

//
std::vector<uint32_t> hgcal::backend::buildCaptureBlockHeader(uint32_t bc,
                                                              uint32_t ec,
                                                              uint32_t oc,
                                                              std::vector<uint8_t>& econdStatus) {
  std::vector<uint32_t> header(2, 0);
  header[0] = (bc & hgcal::BACKEND_FRAME::CAPTUREBLOCK_BC_MASK) << hgcal::BACKEND_FRAME::CAPTUREBLOCK_BC_POS |
              (ec & hgcal::BACKEND_FRAME::CAPTUREBLOCK_EC_MASK) << hgcal::BACKEND_FRAME::CAPTUREBLOCK_EC_POS |
              (oc & hgcal::BACKEND_FRAME::CAPTUREBLOCK_OC_MASK) << hgcal::BACKEND_FRAME::CAPTUREBLOCK_OC_POS |
              (econdStatus[11] & 0x7) << 1 | ((econdStatus[10] >> 2) & 0x1);

  for (size_t i = 0; i < 11; i++) {
    header[1] |= (econdStatus[i] & 0x7) << i * 3;
  }

  return header;
}

//
std::vector<uint32_t> hgcal::backend::buildSlinkHeader(
    uint8_t boe, uint8_t v, uint8_t r8, uint64_t global_event_id, uint8_t r6, uint32_t content_id, uint32_t fed_id) {
  std::vector<uint32_t> header(4, 0);
  header[0] = (boe & hgcal::BACKEND_FRAME::SLINK_BOE_MASK) << hgcal::BACKEND_FRAME::SLINK_BOE_POS |
              (v & hgcal::BACKEND_FRAME::SLINK_V_MASK) << hgcal::BACKEND_FRAME::SLINK_V_POS |
              (r8 & hgcal::BACKEND_FRAME::SLINK_R8_MASK) << hgcal::BACKEND_FRAME::SLINK_R8_POS |
              ((global_event_id >> 41) & SLINK_GLOBAL_EVENTID_MSB_MASK)
                  << hgcal::BACKEND_FRAME::SLINK_GLOBAL_EVENTID_MSB_POS;
  header[1] = (global_event_id & SLINK_GLOBAL_EVENTID_LSB_MASK);
  header[2] = (r6 & hgcal::BACKEND_FRAME::SLINK_R6_MASK) << hgcal::BACKEND_FRAME::SLINK_R6_POS |
              (content_id & hgcal::BACKEND_FRAME::SLINK_CONTENTID_MASK) << hgcal::BACKEND_FRAME::SLINK_CONTENTID_POS;
  header[3] = (fed_id & hgcal::BACKEND_FRAME::SLINK_SOURCEID_MASK) << hgcal::BACKEND_FRAME::SLINK_SOURCEID_POS;

  return header;
}

//
std::vector<uint32_t> hgcal::backend::buildSlinkTrailer(uint8_t eoe,
                                                        uint8_t daqcrc,
                                                        uint8_t trailer_r,
                                                        uint64_t event_length,
                                                        uint8_t bxid,
                                                        uint32_t orbit_id,
                                                        uint32_t crc,
                                                        uint32_t status) {
  std::vector<uint32_t> trailer(4, 0);

  trailer[0] = (eoe & hgcal::BACKEND_FRAME::SLINK_EOE_MASK) << hgcal::BACKEND_FRAME::SLINK_EOE_POS |
               (daqcrc & hgcal::BACKEND_FRAME::SLINK_DAQCRC_MASK) << hgcal::BACKEND_FRAME::SLINK_DAQCRC_POS |
               (trailer_r & hgcal::BACKEND_FRAME::SLINK_TRAILERR_MASK) << hgcal::BACKEND_FRAME::SLINK_TRAILERR_POS;
  trailer[1] = (event_length & hgcal::BACKEND_FRAME::SLINK_EVLENGTH_MASK) << hgcal::BACKEND_FRAME::SLINK_EVLENGTH_POS |
               (bxid & hgcal::BACKEND_FRAME::SLINK_BXID_MASK) << hgcal::BACKEND_FRAME::SLINK_BXID_POS;
  trailer[2] = (orbit_id & hgcal::BACKEND_FRAME::SLINK_ORBID_MASK) << hgcal::BACKEND_FRAME::SLINK_ORBID_POS;
  trailer[3] = (crc & hgcal::BACKEND_FRAME::SLINK_CRC_MASK) << hgcal::BACKEND_FRAME::SLINK_CRC_POS |
               (status & hgcal::BACKEND_FRAME::SLINK_STATUS_MASK) << hgcal::BACKEND_FRAME::SLINK_STATUS_POS;

  return trailer;
}
