#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;
using namespace sipixelobjects;
using namespace sipixelconstants;

PixelDataFormatter::PixelDataFormatter(const SiPixelFedCablingTree* map, bool phase)
    : theDigiCounter_(0),
      theWordCounter_(0),
      theCablingTree_(map),
      badPixelInfo_(nullptr),
      modulesToUnpack_(nullptr),
      phase1_(phase) {
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8 = sizeof(char);
  if (s8 != 1 || s32 != 4 * s8 || s64 != 2 * s32) {
    LogError("UnexpectedSizes") << " unexpected sizes: "
                                << "  size of char is: " << s8 << ", size of Word32 is: " << s32
                                << ", size of Word64 is: " << s64 << ", send exception";
  }
  includeErrors_ = false;
  useQualityInfo_ = false;
  allDetDigis_ = 0;
  hasDetDigis_ = 0;

  if (phase1_) {
    maxROCIndex_ = 8;
    errorcheck_ = std::unique_ptr<ErrorCheckerBase>(new ErrorChecker());
  } else {
    maxROCIndex_ = 25;
    errorcheck_ = std::unique_ptr<ErrorCheckerBase>(new ErrorCheckerPhase0());
  }
}

void PixelDataFormatter::setErrorStatus(bool ErrorStatus) {
  includeErrors_ = ErrorStatus;
  errorcheck_->setErrorStatus(includeErrors_);
}

void PixelDataFormatter::setQualityStatus(bool QualityStatus, const SiPixelQuality* QualityInfo) {
  useQualityInfo_ = QualityStatus;
  badPixelInfo_ = QualityInfo;
}

void PixelDataFormatter::setModulesToUnpack(const std::set<unsigned int>* moduleIds) { modulesToUnpack_ = moduleIds; }

void PixelDataFormatter::passFrameReverter(const SiPixelFrameReverter* reverter) { theFrameReverter_ = reverter; }

void PixelDataFormatter::interpretRawData(
    bool& errorsInEvent, int fedId, const FEDRawData& rawData, Collection& digis, Errors& errors) {
  using namespace sipixelobjects;

  int nWords = rawData.size() / sizeof(Word64);
  if (nWords == 0)
    return;

  SiPixelFrameConverter converter(theCablingTree_, fedId);

  // check CRC bit
  const Word64* trailer = reinterpret_cast<const Word64*>(rawData.data()) + (nWords - 1);
  if (!errorcheck_->checkCRC(errorsInEvent, fedId, trailer, errors))
    return;

  // check headers
  const Word64* header = reinterpret_cast<const Word64*>(rawData.data());
  header--;
  bool moreHeaders = true;
  while (moreHeaders) {
    header++;
    LogTrace("") << "HEADER:  " << print(*header);
    bool headerStatus = errorcheck_->checkHeader(errorsInEvent, fedId, header, errors);
    moreHeaders = headerStatus;
  }

  // check trailers
  bool moreTrailers = true;
  trailer++;
  while (moreTrailers) {
    trailer--;
    LogTrace("") << "TRAILER: " << print(*trailer);
    bool trailerStatus = errorcheck_->checkTrailer(errorsInEvent, fedId, nWords, trailer, errors);
    moreTrailers = trailerStatus;
  }

  // data words
  theWordCounter_ += 2 * (nWords - 2);
  LogTrace("") << "data words: " << (trailer - header - 1);

  int link = -1;
  int roc = -1;
  int layer = 0;
  PixelROC const* rocp = nullptr;
  bool skipROC = false;
  edm::DetSet<PixelDigi>* detDigis = nullptr;

  const Word32* bw = (const Word32*)(header + 1);
  const Word32* ew = (const Word32*)(trailer);
  if (*(ew - 1) == 0) {
    ew--;
    theWordCounter_--;
  }
  for (auto word = bw; word < ew; ++word) {
    LogTrace("") << "DATA: " << print(*word);

    auto ww = *word;
    if UNLIKELY (ww == 0) {
      theWordCounter_--;
      continue;
    }
    int nlink = getLink(ww);
    int nroc = getROC(ww);

    if ((nlink != link) | (nroc != roc)) {  // new roc
      link = nlink;
      roc = nroc;
      skipROC = LIKELY(roc < maxROCIndex_)
                    ? false
                    : !errorcheck_->checkROC(errorsInEvent, fedId, &converter, theCablingTree_, ww, errors);
      if (skipROC)
        continue;
      rocp = converter.toRoc(link, roc);
      if UNLIKELY (!rocp) {
        errorsInEvent = true;
        errorcheck_->conversionError(fedId, &converter, 2, ww, errors);
        skipROC = true;
        continue;
      }
      auto rawId = rocp->rawId();
      bool barrel = PixelModuleName::isBarrel(rawId);
      if (barrel)
        layer = PixelROC::bpixLayerPhase1(rawId);
      else
        layer = 0;

      if (useQualityInfo_ & (nullptr != badPixelInfo_)) {
        short rocInDet = (short)rocp->idInDetUnit();
        skipROC = badPixelInfo_->IsRocBad(rawId, rocInDet);
        if (skipROC)
          continue;
      }
      skipROC = modulesToUnpack_ && (modulesToUnpack_->find(rawId) == modulesToUnpack_->end());
      if (skipROC)
        continue;

      detDigis = &digis.find_or_insert(rawId);
      if ((*detDigis).empty())
        (*detDigis).data.reserve(32);  // avoid the first relocations
    }

    // skip is roc to be skipped ot invalid
    if UNLIKELY (skipROC || !rocp)
      continue;

    int adc = getADC(ww);
    std::unique_ptr<LocalPixel> local;

    if (phase1_ && layer == 1) {  // special case for layer 1ROC
      // for l1 roc use the roc column and row index instead of dcol and pixel index.
      int col = getCol(ww);
      int row = getRow(ww);

      LocalPixel::RocRowCol localCR = {row, col};  // build pixel
      if UNLIKELY (!localCR.valid()) {
        LogDebug("PixelDataFormatter::interpretRawData") << "status #3";
        errorsInEvent = true;
        errorcheck_->conversionError(fedId, &converter, 3, ww, errors);
        continue;
      }
      local = std::make_unique<LocalPixel>(localCR);  // local pixel coordinate

    } else {  // phase0 and phase1 except bpix layer 1
      int dcol = getDCol(ww);
      int pxid = getPxId(ww);
      LocalPixel::DcolPxid localDP = {dcol, pxid};

      if UNLIKELY (!localDP.valid()) {
        LogDebug("PixelDataFormatter::interpretRawData") << "status #3";
        errorsInEvent = true;
        errorcheck_->conversionError(fedId, &converter, 3, ww, errors);
        continue;
      }
      local = std::make_unique<LocalPixel>(localDP);  // local pixel coordinate
    }

    GlobalPixel global = rocp->toGlobal(*local);  // global pixel coordinate (in module)
    (*detDigis).data.emplace_back(global.row, global.col, adc);
    LogTrace("") << (*detDigis).data.back();
  }
}

void PixelDataFormatter::formatRawData(unsigned int lvl1_ID,
                                       RawData& fedRawData,
                                       const Digis& digis,
                                       const BadChannels& badChannels) {
  std::map<int, vector<Word32> > words;

  // translate digis into 32-bit raw words and store in map indexed by Fed
  for (Digis::const_iterator im = digis.begin(); im != digis.end(); im++) {
    allDetDigis_++;
    cms_uint32_t rawId = im->first;
    int layer = 0;
    bool barrel = PixelModuleName::isBarrel(rawId);
    if (barrel)
      layer = PixelROC::bpixLayerPhase1(rawId);

    BadChannels::const_iterator detBadChannels = badChannels.find(rawId);

    hasDetDigis_++;
    const DetDigis& detDigis = im->second;
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); it++) {
      theDigiCounter_++;
      const PixelDigi& digi = (*it);
      int fedId = 0;

      if (layer == 1 && phase1_)
        fedId = digi2wordPhase1Layer1(rawId, digi, words);
      else
        fedId = digi2word(rawId, digi, words);

      if (fedId < 0) {
        LogError("FormatDataException") << " digi2word returns error #" << fedId << " Ndigis: " << theDigiCounter_
                                        << endl
                                        << " detector: " << rawId << endl
                                        << print(digi) << endl;
      } else if (detBadChannels != badChannels.end()) {
        auto badChannel =
            std::find_if(detBadChannels->second.begin(), detBadChannels->second.end(), [&](const PixelFEDChannel& ch) {
              return (int(ch.fed) == fedId && ch.link == getLink(words[fedId].back()));
            });
        if (badChannel != detBadChannels->second.end()) {
          LogError("FormatDataException") << " while marked bad, found digi for FED " << fedId << " Link "
                                          << getLink(words[fedId].back()) << " on module " << rawId << endl
                                          << print(digi) << endl;
        }
      }  // if (fedId)
    }    // for (DetDigis
  }      // for (Digis
  LogTrace(" allDetDigis_/hasDetDigis_ : ") << allDetDigis_ << "/" << hasDetDigis_;

  // fill FED error 25 words
  for (const auto& detBadChannels : badChannels) {
    for (const auto& badChannel : detBadChannels.second) {
      unsigned int FEDError25 = 25;
      Word32 word = (badChannel.link << LINK_shift) | (FEDError25 << ROC_shift);
      words[badChannel.fed].push_back(word);
      theWordCounter_++;
    }
  }

  typedef std::map<int, vector<Word32> >::const_iterator RI;
  for (RI feddata = words.begin(); feddata != words.end(); feddata++) {
    int fedId = feddata->first;
    // since raw words are written in the form of 64-bit packets
    // add extra 32-bit word to make number of words even if necessary
    if (words.find(fedId)->second.size() % 2 != 0)
      words[fedId].push_back(Word32(0));

    // size in Bytes; create output structure
    int dataSize = words.find(fedId)->second.size() * sizeof(Word32);
    int nHeaders = 1;
    int nTrailers = 1;
    dataSize += (nHeaders + nTrailers) * sizeof(Word64);
    FEDRawData* rawData = new FEDRawData(dataSize);

    // get begining of data;
    Word64* word = reinterpret_cast<Word64*>(rawData->data());

    // write one header
    FEDHeader::set(reinterpret_cast<unsigned char*>(word), 0, lvl1_ID, 0, fedId);
    word++;

    // write data
    unsigned int nWord32InFed = words.find(fedId)->second.size();
    for (unsigned int i = 0; i < nWord32InFed; i += 2) {
      *word = (Word64(words.find(fedId)->second[i + 1]) << 32) | words.find(fedId)->second[i];
      LogDebug("PixelDataFormatter") << print(*word);
      word++;
    }

    // write one trailer
    FEDTrailer::set(reinterpret_cast<unsigned char*>(word), dataSize / sizeof(Word64), 0, 0, 0);
    word++;

    // check memory
    if (word != reinterpret_cast<Word64*>(rawData->data() + dataSize)) {
      string s = "** PROBLEM in PixelDataFormatter !!!";
      throw cms::Exception(s);
    }  // if (word !=
    fedRawData[fedId] = *rawData;
    delete rawData;
  }  // for (RI feddata
}

int PixelDataFormatter::digi2word(cms_uint32_t detId,
                                  const PixelDigi& digi,
                                  std::map<int, vector<Word32> >& words) const {
  LogDebug("PixelDataFormatter") << print(digi);

  DetectorIndex detector = {detId, digi.row(), digi.column()};
  ElectronicIndex cabling;
  int fedId = theFrameReverter_->toCabling(cabling, detector);
  if (fedId < 0)
    return fedId;

  Word32 word = (cabling.link << LINK_shift) | (cabling.roc << ROC_shift) | (cabling.dcol << DCOL_shift) |
                (cabling.pxid << PXID_shift) | (digi.adc() << ADC_shift);
  words[fedId].push_back(word);
  theWordCounter_++;

  return fedId;
}
int PixelDataFormatter::digi2wordPhase1Layer1(cms_uint32_t detId,
                                              const PixelDigi& digi,
                                              std::map<int, vector<Word32> >& words) const {
  LogDebug("PixelDataFormatter") << print(digi);

  DetectorIndex detector = {detId, digi.row(), digi.column()};
  ElectronicIndex cabling;
  int fedId = theFrameReverter_->toCabling(cabling, detector);
  if (fedId < 0)
    return fedId;

  int col = ((cabling.dcol) * 2) + ((cabling.pxid) % 2);
  int row = LocalPixel::numRowsInRoc - ((cabling.pxid) / 2);

  Word32 word = (cabling.link << LINK_shift) | (cabling.roc << ROC_shift) | (col << COL_shift) | (row << ROW_shift) |
                (digi.adc() << ADC_shift);
  words[fedId].push_back(word);
  theWordCounter_++;

  return fedId;
}

std::string PixelDataFormatter::print(const PixelDigi& digi) const {
  ostringstream str;
  str << " DIGI: row: " << digi.row() << ", col: " << digi.column() << ", adc: " << digi.adc();
  return str.str();
}

std::string PixelDataFormatter::print(const Word64& word) const {
  ostringstream str;
  str << "word64:  " << reinterpret_cast<const bitset<64>&>(word);
  return str.str();
}

void PixelDataFormatter::unpackFEDErrors(PixelDataFormatter::Errors const& errors,
                                         std::vector<int> const& tkerrorlist,
                                         std::vector<int> const& usererrorlist,
                                         edm::DetSetVector<SiPixelRawDataError>& errorcollection,
                                         DetIdCollection& tkerror_detidcollection,
                                         DetIdCollection& usererror_detidcollection,
                                         edmNew::DetSetVector<PixelFEDChannel>& disabled_channelcollection,
                                         DetErrors& nodeterrors) {
  const uint32_t dummyDetId = 0xffffffff;
  for (const auto& [errorDetId, rawErrorsVec] : errors) {
    if (errorDetId == dummyDetId) {  // errors given dummy detId must be sorted by Fed
      nodeterrors.insert(nodeterrors.end(), rawErrorsVec.begin(), rawErrorsVec.end());
    } else {
      edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection.find_or_insert(errorDetId);
      errorDetSet.data.insert(errorDetSet.data.end(), rawErrorsVec.begin(), rawErrorsVec.end());
      // Fill detid of the detectors where there is error AND the error number is listed
      // in the configurable error list in the job option cfi.
      // Code needs to be here, because there can be a set of errors for each
      // entry in the for loop over PixelDataFormatter::Errors

      std::vector<PixelFEDChannel> disabledChannelsDetSet;

      for (auto const& aPixelError : errorDetSet) {
        // For the time being, we extend the error handling functionality with ErrorType 25
        // In the future, we should sort out how the usage of tkerrorlist can be generalized
        if (phase1_ && aPixelError.getType() == 25) {
          int fedId = aPixelError.getFedId();
          const sipixelobjects::PixelFEDCabling* fed = theCablingTree_->fed(fedId);
          if (fed) {
            cms_uint32_t linkId = getLink(aPixelError.getWord32());
            const sipixelobjects::PixelFEDLink* link = fed->link(linkId);
            if (link) {
              // The "offline" 0..15 numbering is fixed by definition, also, the FrameConversion depends on it
              // in contrast, the ROC-in-channel numbering is determined by hardware --> better to use the "offline" scheme
              PixelFEDChannel ch = {fed->id(), linkId, 25, 0};
              for (unsigned int iRoc = 1; iRoc <= link->numberOfROCs(); iRoc++) {
                const sipixelobjects::PixelROC* roc = link->roc(iRoc);
                if (roc->idInDetUnit() < ch.roc_first)
                  ch.roc_first = roc->idInDetUnit();
                if (roc->idInDetUnit() > ch.roc_last)
                  ch.roc_last = roc->idInDetUnit();
              }
              disabledChannelsDetSet.push_back(ch);
            }
          }
        } else {
          // fill list of detIds to be turned off by tracking
          if (!tkerrorlist.empty()) {
            auto it_find = std::find(tkerrorlist.begin(), tkerrorlist.end(), aPixelError.getType());
            if (it_find != tkerrorlist.end()) {
              tkerror_detidcollection.push_back(errorDetId);
            }
          }
        }

        // fill list of detIds with errors to be studied
        if (!usererrorlist.empty()) {
          auto it_find = std::find(usererrorlist.begin(), usererrorlist.end(), aPixelError.getType());
          if (it_find != usererrorlist.end()) {
            usererror_detidcollection.push_back(errorDetId);
          }
        }

      }  // loop on DetSet of errors

      if (!disabledChannelsDetSet.empty()) {
        disabled_channelcollection.insert(errorDetId, disabledChannelsDetSet.data(), disabledChannelsDetSet.size());
      }

    }  // if error assigned to a real DetId
  }    // loop on errors in event for this FED
}
