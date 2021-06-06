#ifndef UCTDAQRawData_hh
#define UCTDAQRawData_hh

#include <cstdint>
#include <iostream>
#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

class UCTDAQRawData {
public:
  UCTDAQRawData(const uint64_t *d) : myDataPtr(d) {
    if (d != nullptr) {
      if ((d[0] & 0x5000000000000000) != 0x5000000000000000) {
        edm::LogError("UCTDAQRawData") << "CDF Header does not seem to be correct" << std::showbase << std::internal
                                       << std::setfill('0') << std::setw(10) << std::hex << d[0] << "; but continuing!"
                                       << std::endl;
      }
    }
  }

  virtual ~UCTDAQRawData() { ; }

  // Access functions for convenience

  const uint64_t *dataPtr() const { return myDataPtr; }

  const uint64_t *cdfHeader() const { return &myDataPtr[0]; }

  uint32_t FOV() { return ((myDataPtr[0] & 0x00000000000000F0) >> 4); }
  uint32_t sourceID() { return ((myDataPtr[0] & 0x00000000000FFF00) >> 8); }
  uint32_t BXID() { return ((myDataPtr[0] & 0x00000000FFF00000) >> 20); }
  uint32_t L1ID() { return ((myDataPtr[0] & 0x00FFFFFF00000000) >> 32); }
  uint32_t eventType() { return ((myDataPtr[0] & 0x0F00000000000000) >> 56); }
  uint32_t orbitNumber() { return ((myDataPtr[1] & 0x0000000FFFFFFFF0) >> 4); }
  uint32_t nAMCs() { return ((myDataPtr[1] & 0x00F0000000000000) >> 52); }
  uint32_t uFOV() { return ((myDataPtr[1] & 0xF000000000000000) >> 60); }

  uint32_t boardID(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return myDataPtr[2 + amc] & 0x000000000000FFFF;
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch board ID for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return 0xDEADBEEF;
  }

  uint32_t amcNo(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x00000000000F0000) >> 16);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch amc no for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return 0xDEADBEEF;
  }

  uint32_t amcBlkNo(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x000000000FF00000) >> 20);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch amc block no for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return 0xDEADBEEF;
  }

  uint32_t amcSize(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x00FFFFFF00000000) >> 32);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch amc size for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return 0xDEADBEEF;
  }

  bool crcError(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x0100000000000000) != 0);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch crcError-bit for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return false;
  }

  bool isValid(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x0200000000000000) != 0);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch isValid-bit for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return false;
  }

  bool isPresent(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x0400000000000000) != 0);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch isPresent-bit for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return false;
  }

  bool isEnabled(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x0800000000000000) != 0);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch isEnabled-bit for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return false;
  }

  bool isSegmented(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x1000000000000000) != 0);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch isSegmented-bit for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return false;
  }

  bool more(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x2000000000000000) != 0);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch more-bit for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return false;
  }

  bool lengthError(uint32_t amc = 0) {
    if (amc < nAMCs()) {
      return ((myDataPtr[2 + amc] & 0x4000000000000000) != 0);
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch length error bit for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return false;
  }

  const uint32_t *amcPayload(uint32_t amc) {
    if (amc < nAMCs()) {
      // Number of 64-bit words to skip
      uint32_t skip = 2 + nAMCs();
      for (uint32_t i = 0; i < amc; i++) {
        skip += amcSize(i);
      }
      return (uint32_t *)&myDataPtr[skip];
    }
    edm::LogError("UCTDAQRawData") << "UCTDAQRawData: Failed to fetch payload location for AMC = " << amc
                                   << "; Max AMC = " << nAMCs() << std::endl;
    return nullptr;
  }

  const uint64_t *amc13TrailerPtr() {
    uint32_t skip = 2;
    for (uint32_t i = 0; i < nAMCs(); i++) {
      skip += (1 + amcSize(i));
    }
    return &myDataPtr[skip];
  }

  uint32_t amc13BXID() {
    const uint64_t *data = amc13TrailerPtr();
    return (data[0] & 0x0000000000000FFF);
  }

  uint32_t amc13L1ID() {
    const uint64_t *data = amc13TrailerPtr();
    return ((data[0] & 0x00000000000FF000) >> 12);
  }

  uint32_t amc13BlockNo() {
    const uint64_t *data = amc13TrailerPtr();
    return ((data[0] & 0x000000000FF00000) >> 20);
  }

  uint32_t amc13CRC32() {
    const uint64_t *data = amc13TrailerPtr();
    return ((data[0] & 0xFFFFFFFF00000000) >> 32);
  }

  const uint64_t *cdfTrailerPtr() {
    uint32_t skip = 2;
    for (uint32_t i = 0; i < nAMCs(); i++) {
      skip += (1 + amcSize(i));
    }
    skip++;
    const uint64_t *data = &myDataPtr[skip];
    if ((data[0] & 0xF000000000000000) != 0xA000000000000000) {
      edm::LogError("UCTDAQRawData") << "CDF Trailer seems to be wrong : " << std::showbase << std::internal
                                     << std::setfill('0') << std::setw(10) << std::hex << data[1] << std::endl;
    }
    return data;
  }

  bool crcModified() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x0000000000000004) != 0);
  }

  bool isLastTrailerWord() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x0000000000000008) != 0);
  }

  uint32_t ttsBits() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x00000000000000F0) >> 4);
  }

  uint32_t eventStatus() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x0000000000000F00) >> 8);
  }

  bool isWrongFEDID() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x0000000000004000) != 0);
  }

  bool isSLinkErrorDetectedByFRL() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x0000000000008000) != 0);
  }

  uint32_t crc16() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x00000000FFFF0000) >> 16);
  }

  uint32_t eventLength() {
    const uint64_t *data = cdfTrailerPtr();
    return ((data[0] & 0x00FFFFFF00000000) >> 32);
  }

  void print() {
    using namespace std;
    LogDebug("UCTDAQRawData") << "Common cDAQ/AMC13 Data Header:" << endl;
    LogDebug("UCTDAQRawData") << "Framework Version = " << internal << setfill('0') << setw(3) << hex << FOV() << endl;
    LogDebug("UCTDAQRawData") << "sourceID......... = " << dec << sourceID() << endl;
    LogDebug("UCTDAQRawData") << "BXID............. = " << dec << BXID() << endl;
    LogDebug("UCTDAQRawData") << "L1ID............. = " << internal << setfill('0') << setw(8) << hex << L1ID() << endl;
    LogDebug("UCTDAQRawData") << "eventType........ = " << internal << setfill('0') << setw(3) << hex << eventType()
                              << endl;
    LogDebug("UCTDAQRawData") << "orbitNo.......... = " << dec << orbitNumber() << endl;
    LogDebug("UCTDAQRawData") << "uFOV............. = " << internal << setfill('0') << setw(8) << hex << uFOV() << endl;
    LogDebug("UCTDAQRawData") << "# of CTP7s....... = " << dec << nAMCs() << endl;
    LogDebug("UCTDAQRawData")
        << "Phi SlotNo BlockNo     Size CRC? Valid? Present? Enabled? Segmented? More? LengthError?" << endl;
    for (uint32_t i = 0; i < nAMCs(); i++) {
      LogDebug("UCTDAQRawData") << dec << setfill(' ') << setw(3) << boardID(i) << " " << dec << setfill(' ') << setw(6)
                                << amcNo(i) << " " << dec << setfill(' ') << setw(7) << amcBlkNo(i) << " " << dec
                                << setfill(' ') << setw(8) << amcSize(i) << " "
                                << "   " << crcError(i) << " "
                                << "     " << isValid(i) << " "
                                << "       " << isPresent(i) << " "
                                << "       " << isEnabled(i) << " "
                                << "         " << isSegmented(i) << " "
                                << "    " << more(i) << " "
                                << "           " << lengthError(i) << endl;
    }
    LogDebug("UCTDAQRawData") << "AMC13 Trailer:" << endl;
    LogDebug("UCTDAQRawData") << "AMC13 BXID....... = " << dec << amc13BXID() << endl;
    LogDebug("UCTDAQRawData") << "AMC13 L1ID....... = " << dec << amc13L1ID() << endl;
    LogDebug("UCTDAQRawData") << "AMC13 BlockNo.... = " << dec << amc13BlockNo() << endl;
    LogDebug("UCTDAQRawData") << "AMC13 CRC32...... = " << internal << setfill('0') << setw(10) << hex << amc13BXID()
                              << endl;
    LogDebug("UCTDAQRawData") << "Common DAQ Trailer:" << endl;
    LogDebug("UCTDAQRawData") << "CRC Modified?.... = " << crcModified() << endl;
    LogDebug("UCTDAQRawData") << "Last Trailer Word?= " << isLastTrailerWord() << endl;
    LogDebug("UCTDAQRawData") << "ttsBits.......... = " << internal << setfill('0') << setw(3) << hex << ttsBits()
                              << endl;
    LogDebug("UCTDAQRawData") << "Event Status..... = " << internal << setfill('0') << setw(3) << hex << eventStatus()
                              << endl;
    LogDebug("UCTDAQRawData") << "Wrong FED ID?.... = " << isWrongFEDID() << endl;
    LogDebug("UCTDAQRawData") << "SLink Error?..... = " << isSLinkErrorDetectedByFRL() << endl;
    LogDebug("UCTDAQRawData") << "CRC 16........... = " << internal << setfill('0') << setw(6) << hex << crc16()
                              << endl;
    LogDebug("UCTDAQRawData") << "Event Length..... = " << dec << eventLength() << endl;
  }

private:
  // No copy constructor and equality operator are needed

  UCTDAQRawData(const UCTDAQRawData &) = delete;
  const UCTDAQRawData &operator=(const UCTDAQRawData &i) = delete;

  // RawData data

  const uint64_t *myDataPtr;
};

#endif
