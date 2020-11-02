/** \unpacker for gem
 *  \author J. Lee, Yechan Kang - UoS
 */
#include "EventFilter/GEMRawToDigi/interface/GEMRawToDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace gem;

std::unique_ptr<AMC13Event> GEMRawToDigi::convertWordToAMC13Event(const uint64_t* word) {
  vfatError_ = false;
  amcError_ = false;

  auto amc13Event = std::make_unique<AMC13Event>();

  amc13Event->setCDFHeader(*word);
  amc13Event->setAMC13Header(*(++word));

  // Readout out AMC headers
  for (uint8_t i = 0; i < amc13Event->nAMC(); ++i)
    amc13Event->addAMCheader(*(++word));

  // Readout out AMC payloads
  for (uint8_t i = 0; i < amc13Event->nAMC(); ++i) {
    auto amcData = AMCdata();
    amcData.setAMCheader1(*(++word));
    amcData.setAMCheader2(*(++word));
    amcData.setGEMeventHeader(*(++word));

    // Fill GEB
    for (uint8_t j = 0; j < amcData.davCnt(); ++j) {
      auto gebData = GEBdata();
      gebData.setChamberHeader(*(++word));

      // Fill vfat
      for (uint16_t k = 0; k < gebData.vfatWordCnt() / 3; k++) {
        auto vfatData = VFATdata();
        vfatData.read_fw(*(++word));
        vfatData.read_sw(*(++word));
        vfatData.read_tw(*(++word));
        gebData.addVFAT(vfatData);

      }  // end of vfat loop

      gebData.setChamberTrailer(*(++word));
      if (gebData.vfatWordCnt() != gebData.vfatWordCntT()) {
        vfatError_ = true;
      }
      amcData.addGEB(gebData);

    }  // end of geb loop

    amcData.setGEMeventTrailer(*(++word));
    amcData.setAMCTrailer(*(++word));
    if (amc13Event->getAMCsize(i) != amcData.dataLength()) {
      amcError_ = true;
    }
    amc13Event->addAMCpayload(amcData);

  }  // end of amc loop

  amc13Event->setAMC13Trailer(*(++word));
  amc13Event->setCDFTrailer(*(++word));

  return amc13Event;
}
