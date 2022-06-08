/** \unpacker for gem
 *  \author J. Lee, Yechan Kang - UoS
 */
#include "EventFilter/GEMRawToDigi/interface/GEMRawToDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::unique_ptr<GEMAMC13> GEMRawToDigi::convertWordToGEMAMC13(const uint64_t* word) {
  vfatError_ = false;
  amcError_ = false;

  auto amc13 = std::make_unique<GEMAMC13>();

  amc13->setCDFHeader(*word);
  amc13->setAMC13Header(*(++word));

  // Readout out AMC headers
  for (uint8_t i = 0; i < amc13->nAMC(); ++i)
    amc13->addAMCheader(*(++word));

  // Readout out AMC payloads
  for (uint8_t i = 0; i < amc13->nAMC(); ++i) {
    auto amc = GEMAMC();
    amc.setAMCheader1(*(++word));
    amc.setAMCheader2(*(++word));
    amc.setGEMeventHeader(*(++word));

    // Fill GEB
    for (uint8_t j = 0; j < amc.davCnt(); ++j) {
      auto oh = GEMOptoHybrid();
      oh.setVersion(amc.formatVer());
      oh.setChamberHeader(*(++word));

      // Fill vfat
      for (uint16_t k = 0; k < oh.vfatWordCnt() / 3; k++) {
        auto vfat = GEMVFAT();
        vfat.read_fw(*(++word));
        vfat.read_sw(*(++word));
        vfat.read_tw(*(++word));
        oh.addVFAT(vfat);

      }  // end of vfat loop

      oh.setChamberTrailer(*(++word));
      amc.addGEB(oh);

    }  // end of geb loop

    amc.setGEMeventTrailer(*(++word));
    amc.setAMCTrailer(*(++word));
    amc13->addAMCpayload(amc);

  }  // end of amc loop

  amc13->setAMC13Trailer(*(++word));
  amc13->setCDFTrailer(*(++word));

  return amc13;
}
