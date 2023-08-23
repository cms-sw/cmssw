#include "EventFilter/CTPPSRawToDigi/interface/CTPPSPixelDataFormatter.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "CondFormats/PPSObjects/interface/CTPPSPixelROC.h"  //KS

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace edm;
using namespace std;

namespace {
  constexpr int m_LINK_bits = 6;
  constexpr int m_ROC_bits = 5;
  constexpr int m_DCOL_bits = 5;
  constexpr int m_PXID_bits = 8;
  constexpr int m_COL_bits = 6;
  constexpr int m_ROW_bits = 7;
  constexpr int m_ADC_bits = 8;
  constexpr int min_Dcol = 0;
  constexpr int max_Dcol = 25;
  constexpr int min_Pixid = 2;
  constexpr int max_Pixid = 161;
  constexpr int min_COL = 0;
  constexpr int max_COL = 51;
  constexpr int min_ROW = 0;
  constexpr int max_ROW = 79;
  constexpr int maxRocIndex = 3;
  constexpr int maxLinkIndex = 49;

}  // namespace

CTPPSPixelDataFormatter::CTPPSPixelDataFormatter(std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo> const& mapping,
                                                 CTPPSPixelErrorSummary& eSummary)
    : m_WordCounter(0),
      m_Mapping(mapping),
      m_ErrorSummary(eSummary)

{
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8 = sizeof(char);
  if (s8 != 1 || s32 != 4 * s8 || s64 != 2 * s32) {
    LogError("UnexpectedSizes") << " unexpected sizes: "
                                << "  size of char is: " << s8 << ", size of Word32 is: " << s32
                                << ", size of Word64 is: " << s64 << ", send exception";
  }

  m_ADC_shift = 0;
  m_PXID_shift = m_ADC_shift + m_ADC_bits;
  m_DCOL_shift = m_PXID_shift + m_PXID_bits;
  //Run3 shifts
  m_ROW_shift = m_ADC_shift + m_ADC_bits;
  m_COL_shift = m_ROW_shift + m_ROW_bits;

  m_ROC_shift = m_DCOL_shift + m_DCOL_bits;

  m_LINK_shift = m_ROC_shift + m_ROC_bits;
  m_LINK_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_LINK_bits);
  m_ROC_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_ROC_bits);

  m_DCOL_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_DCOL_bits);
  m_PXID_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_PXID_bits);
  //Run3 masks
  m_COL_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_COL_bits);
  m_ROW_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_ROW_bits);

  m_ADC_mask = ~(~CTPPSPixelDataFormatter::Word32(0) << m_ADC_bits);
}

void CTPPSPixelDataFormatter::setErrorStatus(bool errorStatus) {
  m_IncludeErrors = errorStatus;
  m_ErrorCheck.setErrorStatus(m_IncludeErrors);
}

void CTPPSPixelDataFormatter::interpretRawData(
    const bool& isRun3, bool& errorsInEvent, int fedId, const FEDRawData& rawData, Collection& digis, Errors& errors) {
  int nWords = rawData.size() / sizeof(Word64);
  if (nWords == 0)
    return;

  /// check CRC bit
  const Word64* trailer = reinterpret_cast<const Word64*>(rawData.data()) + (nWords - 1);
  if (!m_ErrorCheck.checkCRC(errorsInEvent, fedId, trailer, errors))
    return;

  /// check headers
  const Word64* header = reinterpret_cast<const Word64*>(rawData.data());
  header--;
  bool moreHeaders = true;
  while (moreHeaders) {
    header++;
    LogTrace("") << "HEADER:  " << print(*header);
    bool headerStatus = m_ErrorCheck.checkHeader(errorsInEvent, fedId, header, errors);
    moreHeaders = headerStatus;
  }

  /// check trailers
  bool moreTrailers = true;
  trailer++;
  while (moreTrailers) {
    trailer--;
    LogTrace("") << "TRAILER: " << print(*trailer);
    bool trailerStatus = m_ErrorCheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors);
    moreTrailers = trailerStatus;
  }

  /// data words
  m_WordCounter += 2 * (nWords - 2);
  LogTrace("") << "data words: " << (trailer - header - 1);

  int link = -1;
  int roc = -1;

  bool skipROC = false;

  edm::DetSet<CTPPSPixelDigi>* detDigis = nullptr;

  const Word32* bw = (const Word32*)(header + 1);
  const Word32* ew = (const Word32*)(trailer);
  if (*(ew - 1) == 0) {
    ew--;
    m_WordCounter--;
  }

  for (auto word = bw; word < ew; ++word) {
    LogTrace("") << "DATA: " << print(*word);
    auto ww = *word;
    if UNLIKELY (ww == 0) {
      m_WordCounter--;
      continue;
    }

    int nlink = (ww >> m_LINK_shift) & m_LINK_mask;
    int nroc = (ww >> m_ROC_shift) & m_ROC_mask;
    int FMC = 0;
    uint32_t iD = RPixErrorChecker::dummyDetId;  //0xFFFFFFFF; //dummyDetId
    int convroc = nroc - 1;

    CTPPSPixelROC rocp;
    CTPPSPixelFramePosition fPos(fedId, FMC, nlink, convroc);
    std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo>::const_iterator mit;
    mit = m_Mapping.find(fPos);
    if (mit != m_Mapping.end()) {
      CTPPSPixelROCInfo rocInfo = (*mit).second;
      iD = rocInfo.iD;
      rocp.setParameters(iD, rocInfo.roc, convroc);
    }

    if ((nlink != link) | (nroc != roc)) {  // new roc
      link = nlink;
      roc = nroc;

      if ((roc - 1) < maxRocIndex) {
        skipROC = false;
      } else {
        // using dummy detId - recovering of FED channel foreseen in DQM
        iD = RPixErrorChecker::dummyDetId;
        skipROC = !m_ErrorCheck.checkROC(errorsInEvent, fedId, iD, ww, errors);
      }
      if (skipROC)
        continue;

      if (mit == m_Mapping.end()) {
        if (nlink >= maxLinkIndex) {
          m_ErrorCheck.conversionError(fedId, iD, InvalidLinkId, ww, errors);

          m_ErrorSummary.add("Invalid linkId", "");
        } else if ((nroc - 1) >= maxRocIndex) {
          m_ErrorCheck.conversionError(fedId, iD, InvalidROCId, ww, errors);
          m_ErrorSummary.add("Invalid ROC",
                             fmt::format("Id {0}, in link {1}, of FED {2} in DetId {3}", convroc, nlink, fedId, iD));

        } else {
          m_ErrorCheck.conversionError(fedId, iD, Unknown, ww, errors);
          m_ErrorSummary.add("Error unknown");
        }
        skipROC = true;  // skipping roc due to mapping errors
        continue;
      }
      if (rocp.rawId() == 0) {
        skipROC = true;
        continue;
      }

      auto rawId = rocp.rawId();

      detDigis = &digis.find_or_insert(rawId);
      if ((*detDigis).empty())
        (*detDigis).data.reserve(32);  // avoid the first relocations
    }

    if (skipROC || rocp.rawId() == 0)
      continue;

    int adc = (ww >> m_ADC_shift) & m_ADC_mask;

    int dcol = (ww >> m_DCOL_shift) & m_DCOL_mask;
    int pxid = (ww >> m_PXID_shift) & m_PXID_mask;
    int col = (ww >> m_COL_shift) & m_COL_mask;
    int row = (ww >> m_ROW_shift) & m_ROW_mask;

    if (!isRun3 && (dcol < min_Dcol || dcol > max_Dcol || pxid < min_Pixid || pxid > max_Pixid)) {
      m_ErrorSummary.add(
          "unphysical dcol and/or pxid",
          fmt::format("fedId= {0}, nllink= {1}, convroc= {2}, adc= {3}, dcol= {4}, pxid= {5}, detId= {6}",
                      fedId,
                      nlink,
                      convroc,
                      adc,
                      dcol,
                      pxid,
                      iD));

      m_ErrorCheck.conversionError(fedId, iD, InvalidPixelId, ww, errors);

      continue;
    }
    if (isRun3 && (col < min_COL || col > max_COL || row < min_ROW || row > max_ROW)) {
      m_ErrorSummary.add("unphysical col and/or row",
                         fmt::format("fedId= {0}, nllink= {1}, convroc= {2}, adc= {3}, col= {4}, row= {5}, detId= {6}",
                                     fedId,
                                     nlink,
                                     convroc,
                                     adc,
                                     col,
                                     row,
                                     iD));

      m_ErrorCheck.conversionError(fedId, iD, InvalidPixelId, ww, errors);

      continue;
    }

    std::pair<int, int> rocPixel;
    std::pair<int, int> modPixel;

    if (isRun3) {
      rocPixel = std::make_pair(row, col);
      modPixel = rocp.toGlobal(rocPixel);
    } else {
      rocPixel = std::make_pair(dcol, pxid);
      modPixel = rocp.toGlobalfromDcol(rocPixel);
    }

    CTPPSPixelDigi testdigi(modPixel.first, modPixel.second, adc);

    if (detDigis)
      (*detDigis).data.emplace_back(modPixel.first, modPixel.second, adc);
  }
}

void CTPPSPixelDataFormatter::formatRawData(const bool& isRun3,
                                            unsigned int lvl1_ID,
                                            RawData& fedRawData,
                                            const Digis& digis,
                                            std::vector<PPSPixelIndex> iDdet2fed) {
  std::map<int, vector<Word32> > words;
  // translate digis into 32-bit raw words and store in map indexed by Fed
  m_allDetDigis = 0;
  m_hasDetDigis = 0;
  for (auto const& im : digis) {
    m_allDetDigis++;
    cms_uint32_t rawId = im.first;

    const DetDigis& detDigis = im.second;
    for (auto const& it : detDigis) {
      int nroc = 999, nlink = 999;
      int rocPixelRow = -1, rocPixelColumn = -1, rocID = -1;
      int modulePixelColumn = it.column();
      int modulePixelRow = it.row();

      m_Indices.transformToROC(modulePixelColumn, modulePixelRow, rocID, rocPixelColumn, rocPixelRow);
      const int dcol = m_Indices.DColumn(rocPixelColumn);
      const int pxid = 2 * (rpixValues::ROCSizeInX - rocPixelRow) + (rocPixelColumn % 2);

      unsigned int urocID = rocID;
      PPSPixelIndex myTest = {rawId, urocID, 0, 0, 0};
      // the range has always at most one element
      auto range = std::equal_range(iDdet2fed.begin(), iDdet2fed.end(), myTest, compare);
      if (range.first != range.second) {
        auto i = range.first - iDdet2fed.begin();
        nlink = iDdet2fed.at(i).fedch;
        nroc = iDdet2fed.at(i).rocch + 1;

        pps::pixel::ElectronicIndex cabling = {nlink, nroc, dcol, pxid};
        if (isRun3) {
          cms_uint32_t word = (cabling.link << m_LINK_shift) | (cabling.roc << m_ROC_shift) |
                              (rocPixelColumn << m_COL_shift) | (rocPixelRow << m_ROW_shift) |
                              (it.adc() << m_ADC_shift);

          words[iDdet2fed.at(i).fedid].push_back(word);
        } else {
          cms_uint32_t word = (cabling.link << m_LINK_shift) | (cabling.roc << m_ROC_shift) |
                              (cabling.dcol << m_DCOL_shift) | (cabling.pxid << m_PXID_shift) |
                              (it.adc() << m_ADC_shift);

          words[iDdet2fed.at(i).fedid].push_back(word);
        }
        m_WordCounter++;
        m_hasDetDigis++;

      }  // range
    }    // for DetDigis
  }      // for Digis

  LogTrace(" allDetDigis/hasDetDigis : ") << m_allDetDigis << "/" << m_hasDetDigis;
  for (auto const& feddata : words) {
    int fedId = feddata.first;

    // since raw words are written in the form of 64-bit packets
    // add extra 32-bit word to make number of words even if necessary
    if (words.find(fedId)->second.size() % 2 != 0)
      words[fedId].emplace_back(0);

    // size in Bytes; create output structure
    size_t dataSize = words.find(fedId)->second.size() * sizeof(Word32);
    int nHeaders = 1;
    int nTrailers = 1;
    dataSize += (nHeaders + nTrailers) * sizeof(Word64);

    FEDRawData rawData{dataSize};

    // get begining of data;
    Word64* word = reinterpret_cast<Word64*>(rawData.data());

    // write one header
    FEDHeader::set(reinterpret_cast<unsigned char*>(word), 0, lvl1_ID, 0, fedId);
    word++;

    // write data
    unsigned int nWord32InFed = words.find(fedId)->second.size();
    for (unsigned int i = 0; i < nWord32InFed; i += 2) {
      *word = (Word64(words.find(fedId)->second[i]) << 32) | words.find(fedId)->second[i + 1];
      LogDebug("CTPPSPixelDataFormatter") << print(*word);
      word++;
    }

    // write one trailer
    FEDTrailer::set(reinterpret_cast<unsigned char*>(word), dataSize / sizeof(Word64), 0, 0, 0);
    word++;

    // check memory
    if (word != reinterpret_cast<Word64*>(rawData.data() + dataSize)) {
      //if (word != reinterpret_cast<Word64* >(rawData->data()+dataSize)) {
      string s = "** PROBLEM in CTPPSPixelDataFormatter !!!";
      LogError("CTPPSPixelDataFormatter") << "** PROBLEM in CTPPSPixelDataFormatter!!!";
      throw cms::Exception(s);
    }  // if (word !=
    fedRawData[fedId] = rawData;
  }  // for (RI feddata
}

std::string CTPPSPixelDataFormatter::print(const Word64& word) const {
  std::ostringstream str;
  str << "word64:  " << reinterpret_cast<const std::bitset<64>&>(word);
  return str.str();
}
