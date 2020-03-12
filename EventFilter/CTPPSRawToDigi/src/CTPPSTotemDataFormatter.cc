/****************************************************************************
 *
 * This is a part of PPS offline software.
 * Authors: D.J.Damiao , M.E.Pol
 ****************************************************************************/

#include "EventFilter/CTPPSRawToDigi/interface/CTPPSTotemDataFormatter.h"

#include "EventFilter/CTPPSRawToDigi/interface/CounterChecker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include "EventFilter/CTPPSRawToDigi/interface/DiamondVFATFrame.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"

#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"
#include <cinttypes>
#include <cstdint>

#include <iostream>
#include <iomanip>

using namespace std;
using namespace edm;
typedef uint64_t word;

CTPPSTotemDataFormatter::CTPPSTotemDataFormatter(std::map<TotemFramePosition, TotemVFATInfo> const &mapping)
    : m_WordCounter(0), m_DigiCounter(0) {
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8 = sizeof(char);
  if (s8 != 1 || s32 != 4 * s8 || s64 != 2 * s32) {
    LogError("UnexpectedSizes") << " unexpected sizes: "
                                << "  size of char is: " << s8 << ", size of Word32 is: " << s32
                                << ", size of Word64 is: " << s64 << ", send exception";
  }
}

void CTPPSTotemDataFormatter::formatRawData(unsigned int lvl1_ID,
                                            RawData &fedRawData,
                                            const Digis &digis,
                                            std::vector<PPSStripIndex> iDdet2fed) {
  std::map<int, vector<std::array<uint16_t, 12>>> words;
  for (auto const &itDig : digis) {
    int fedId;
    uint32_t rawId = itDig.first;
    const DetDigis &detDigis = itDig.second;
    std::array<uint16_t, 12> buf, bufCRC;
    std::map<uint32_t, vector<int>> mapIdCh;
    for (auto const &it : detDigis) {
      m_DigiCounter++;
      const int nCH = 128;
      int nStrip = it.stripNumber();
      int chipPosition = nStrip / nCH;
      int channel = nStrip - chipPosition * nCH;        //channel from DIGI
      uint32_t newrawId = rawId + 8192 * chipPosition;  //8192 - distance between chipIds
      mapIdCh[newrawId].push_back(channel);
    }
    for (auto &pId : mapIdCh) {
      PPSStripIndex myTest = {pId.first, 0, 0, 0, 0};

      // the range has always at most one element
      auto range = std::equal_range(iDdet2fed.begin(), iDdet2fed.end(), myTest, compare);
      if (range.first != range.second) {
        auto i = range.first - iDdet2fed.begin();
        for (int b = 0; b < 12; b++) {
          buf[b] = 0;
          bufCRC[b] = 0;
        }
        TotemRPDetId chipId(iDdet2fed.at(i).id);
        for (auto &ch : pId.second) {
          int chInWord = ch / 16;
          buf[9 - chInWord] |= (1 << (ch % 16));  //data[8->1]
          bufCRC[chInWord + 1] = buf[9 - chInWord];
        }
        fedId = iDdet2fed.at(i).fedid;
        int idxInFiber = iDdet2fed.at(i).idxinfiber;
        int gohId = iDdet2fed.at(i).gohid;
        unsigned int hFlag = 0x90;                                 //vmRaw
        buf[0] = (hFlag << 8) | (gohId << 4) | (idxInFiber << 0);  //
        buf[1] = (pId.first >> 15);                                //data[9]
        bufCRC[9] = buf[1];
        Word16 crc_fin = 0xffff;
        for (int i = 11; i >= 1; i--)
          crc_fin = VFATFrame::calculateCRC(crc_fin, bufCRC[i]);
        buf[10] = crc_fin;                            //data[0]
        buf[11] = (15 << 12) | (0 << 8) | (12 << 0);  //build trailer as RawDataUnpacker
        m_WordCounter++;
        words[fedId].push_back(buf);
      }  // range
    }    // mapIdCh
  }      //digis

  typedef std::map<int, vector<std::array<uint16_t, 12>>>::const_iterator RI;
  std::map<int, vector<uint16_t>> words16;
  for (auto &itFed : words) {
    int fedId = itFed.first;
    int wordsS = words.find(fedId)->second.size();
    //due to OrbitCounter block at RawDataUnpacker
    words16[fedId].push_back(Word16(0));
    words16[fedId].push_back(Word16(0));
    //writing data in 16-bit words
    for (int k = 0; k < wordsS; k++) {
      for (int b = 0; b < 12; b++) {
        words16[fedId].push_back(words.find(fedId)->second.at(k)[b]);
      }
    }
    // since raw words are written in the form of 64-bit packets
    // add extra 16-bit words to make number of words even if necessary
    while (words16.find(fedId)->second.size() % 4 != 0)
      words16[fedId].push_back(Word16(0));

    // size in Bytes; create output structure
    auto dataSize = (words16.find(fedId)->second.size()) * sizeof(Word16);
    int nHeaders = 1;
    int nTrailers = 1;
    dataSize += (nHeaders + nTrailers) * sizeof(Word64);

    FEDRawData rawData{dataSize};

    // get begining of data;
    Word64 *word = reinterpret_cast<Word64 *>(rawData.data());

    // write one header
    FEDHeader::set(reinterpret_cast<unsigned char *>(word), 0, lvl1_ID, 0, fedId, 3);
    word++;

    // write data
    unsigned int nWord16InFed = words16.find(fedId)->second.size();
    for (unsigned int i = 0; i < nWord16InFed; i += 4) {
      *word = (Word64(words16.find(fedId)->second[i + 3]) << 48) | (Word64(words16.find(fedId)->second[i + 2]) << 32) |
              (Word64(words16.find(fedId)->second[i + 1]) << 16) | words16.find(fedId)->second[i];
      word++;
    }

    // write one trailer
    FEDTrailer::set(reinterpret_cast<unsigned char *>(word), dataSize / sizeof(Word64), 0, 0, 0);
    word++;

    // check memory
    if (word != reinterpret_cast<Word64 *>(rawData.data() + dataSize)) {
      string s = "** PROBLEM in CTPPSTotemDataFormatter !!!";
      throw cms::Exception(s);
    }
    fedRawData[fedId] = rawData;
  }  // end of rawData
}
