#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;
using namespace sipixelobjects;
namespace {
  constexpr int LINK_bits = 6;
  constexpr int ROC_bits = 5;
  constexpr int DCOL_bits = 5;
  constexpr int PXID_bits = 8;
  constexpr int ADC_bits = 8;

  // Add phase1 constants
  // For phase1
  //GO BACK TO OLD VALUES. THE 48-CHAN FED DOES NOT NEED A NEW FORMAT
  // 28/9/16 d.k.
  constexpr int LINK_bits1 = 6;  // 7;
  constexpr int ROC_bits1 = 5;   // 4;
  // Special for layer 1 bpix rocs 6/9/16 d.k. THIS STAYS.
  constexpr int COL_bits1_l1 = 6;
  constexpr int ROW_bits1_l1 = 7;

  // Moved to the header file, keep commented out unti the final version is done/
  // constexpr int ADC_shift  = 0;
  // constexpr int PXID_shift = ADC_shift + ADC_bits;
  // constexpr int DCOL_shift = PXID_shift + PXID_bits;
  // constexpr int ROC_shift  = DCOL_shift + DCOL_bits;
  // constexpr int LINK_shift = ROC_shift + ROC_bits;
  // constexpr PixelDataFormatter::Word32 LINK_mask = ~(~PixelDataFormatter::Word32(0) << LINK_bits);
  // constexpr PixelDataFormatter::Word32 ROC_mask  = ~(~PixelDataFormatter::Word32(0) << ROC_bits);
  // constexpr PixelDataFormatter::Word32 DCOL_mask = ~(~PixelDataFormatter::Word32(0) << DCOL_bits);
  // constexpr PixelDataFormatter::Word32 PXID_mask = ~(~PixelDataFormatter::Word32(0) << PXID_bits);
  // constexpr PixelDataFormatter::Word32 ADC_mask  = ~(~PixelDataFormatter::Word32(0) << ADC_bits);
  //const bool DANEK = false;
}  // namespace

PixelDataFormatter::PixelDataFormatter(const SiPixelFedCabling* map, bool phase)
    : theDigiCounter(0),
      theWordCounter(0),
      theCablingTree(map),
      badPixelInfo(nullptr),
      modulesToUnpack(nullptr),
      phase1(phase) {
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8 = sizeof(char);
  if (s8 != 1 || s32 != 4 * s8 || s64 != 2 * s32) {
    LogError("UnexpectedSizes") << " unexpected sizes: "
                                << "  size of char is: " << s8 << ", size of Word32 is: " << s32
                                << ", size of Word64 is: " << s64 << ", send exception";
  }
  includeErrors = false;
  useQualityInfo = false;
  allDetDigis = 0;
  hasDetDigis = 0;

  ADC_shift = 0;
  PXID_shift = ADC_shift + ADC_bits;
  DCOL_shift = PXID_shift + PXID_bits;
  ROC_shift = DCOL_shift + DCOL_bits;

  if (phase1) {  // for phase 1
    LINK_shift = ROC_shift + ROC_bits1;
    LINK_mask = ~(~PixelDataFormatter::Word32(0) << LINK_bits1);
    ROC_mask = ~(~PixelDataFormatter::Word32(0) << ROC_bits1);
    // special for layer 1 ROC
    ROW_shift = ADC_shift + ADC_bits;
    COL_shift = ROW_shift + ROW_bits1_l1;
    COL_mask = ~(~PixelDataFormatter::Word32(0) << COL_bits1_l1);
    ROW_mask = ~(~PixelDataFormatter::Word32(0) << ROW_bits1_l1);
    maxROCIndex = 8;

  } else {  // for phase 0
    LINK_shift = ROC_shift + ROC_bits;
    LINK_mask = ~(~PixelDataFormatter::Word32(0) << LINK_bits);
    ROC_mask = ~(~PixelDataFormatter::Word32(0) << ROC_bits);
    maxROCIndex = 25;
  }

  DCOL_mask = ~(~PixelDataFormatter::Word32(0) << DCOL_bits);
  PXID_mask = ~(~PixelDataFormatter::Word32(0) << PXID_bits);
  ADC_mask = ~(~PixelDataFormatter::Word32(0) << ADC_bits);

  if (phase1) {
    errorcheck = std::unique_ptr<ErrorCheckerBase>(new ErrorChecker());
  } else {
    errorcheck = std::unique_ptr<ErrorCheckerBase>(new ErrorCheckerPhase0());
  }
}

void PixelDataFormatter::setErrorStatus(bool ErrorStatus) {
  includeErrors = ErrorStatus;
  errorcheck->setErrorStatus(includeErrors);
}

void PixelDataFormatter::setQualityStatus(bool QualityStatus, const SiPixelQuality* QualityInfo) {
  useQualityInfo = QualityStatus;
  badPixelInfo = QualityInfo;
}

void PixelDataFormatter::setModulesToUnpack(const std::set<unsigned int>* moduleIds) { modulesToUnpack = moduleIds; }

void PixelDataFormatter::passFrameReverter(const SiPixelFrameReverter* reverter) { theFrameReverter = reverter; }

void PixelDataFormatter::interpretRawData(
    bool& errorsInEvent, int fedId, const FEDRawData& rawData, Collection& digis, Errors& errors) {
  using namespace sipixelobjects;

  int nWords = rawData.size() / sizeof(Word64);
  if (nWords == 0)
    return;

  SiPixelFrameConverter converter(theCablingTree, fedId);

  // check CRC bit
  const Word64* trailer = reinterpret_cast<const Word64*>(rawData.data()) + (nWords - 1);
  if (!errorcheck->checkCRC(errorsInEvent, fedId, trailer, errors))
    return;

  // check headers
  const Word64* header = reinterpret_cast<const Word64*>(rawData.data());
  header--;
  bool moreHeaders = true;
  while (moreHeaders) {
    header++;
    LogTrace("") << "HEADER:  " << print(*header);
    bool headerStatus = errorcheck->checkHeader(errorsInEvent, fedId, header, errors);
    moreHeaders = headerStatus;
  }

  // check trailers
  bool moreTrailers = true;
  trailer++;
  while (moreTrailers) {
    trailer--;
    LogTrace("") << "TRAILER: " << print(*trailer);
    bool trailerStatus = errorcheck->checkTrailer(errorsInEvent, fedId, nWords, trailer, errors);
    moreTrailers = trailerStatus;
  }

  // data words
  theWordCounter += 2 * (nWords - 2);
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
    theWordCounter--;
  }
  for (auto word = bw; word < ew; ++word) {
    LogTrace("") << "DATA: " << print(*word);

    auto ww = *word;
    if
      UNLIKELY(ww == 0) {
        theWordCounter--;
        continue;
      }
    int nlink = (ww >> LINK_shift) & LINK_mask;
    int nroc = (ww >> ROC_shift) & ROC_mask;

    //if(DANEK) cout<<" fed, link, roc "<<fedId<<" "<<nlink<<" "<<nroc<<endl;

    if ((nlink != link) | (nroc != roc)) {  // new roc
      link = nlink;
      roc = nroc;
      skipROC = LIKELY(roc < maxROCIndex)
                    ? false
                    : !errorcheck->checkROC(errorsInEvent, fedId, &converter, theCablingTree, ww, errors);
      if (skipROC)
        continue;
      rocp = converter.toRoc(link, roc);
      if
        UNLIKELY(!rocp) {
          errorsInEvent = true;
          errorcheck->conversionError(fedId, &converter, 2, ww, errors);
          skipROC = true;
          continue;
        }
      auto rawId = rocp->rawId();
      bool barrel = PixelModuleName::isBarrel(rawId);
      if (barrel)
        layer = PixelROC::bpixLayerPhase1(rawId);
      else
        layer = 0;

      //if(DANEK) cout<<" rocp "<<rocp->print()<<" layer "<<rocp->bpixLayerPhase1(rawId)<<" "
      //  <<layer<<" phase1 "<<phase1<<" rawid "<<rawId<<endl;

      if (useQualityInfo & (nullptr != badPixelInfo)) {
        short rocInDet = (short)rocp->idInDetUnit();
        skipROC = badPixelInfo->IsRocBad(rawId, rocInDet);
        if (skipROC)
          continue;
      }
      skipROC = modulesToUnpack && (modulesToUnpack->find(rawId) == modulesToUnpack->end());
      if (skipROC)
        continue;

      detDigis = &digis.find_or_insert(rawId);
      if ((*detDigis).empty())
        (*detDigis).data.reserve(32);  // avoid the first relocations
    }

    // skip is roc to be skipped ot invalid
    if
      UNLIKELY(skipROC || !rocp) continue;

    int adc = (ww >> ADC_shift) & ADC_mask;
    std::unique_ptr<LocalPixel> local;

    if (phase1 && layer == 1) {  // special case for layer 1ROC
      // for l1 roc use the roc column and row index instead of dcol and pixel index.
      int col = (ww >> COL_shift) & COL_mask;
      int row = (ww >> ROW_shift) & ROW_mask;
      //if(DANEK) cout<<" layer 1: raw2digi "<<link<<" "<<roc<<" "
      //  <<col<<" "<<row<<" "<<adc<<endl;

      LocalPixel::RocRowCol localCR = {row, col};  // build pixel
      //if(DANEK)cout<<localCR.rocCol<<" "<<localCR.rocRow<<endl;
      if
        UNLIKELY(!localCR.valid()) {
          LogDebug("PixelDataFormatter::interpretRawData") << "status #3";
          errorsInEvent = true;
          errorcheck->conversionError(fedId, &converter, 3, ww, errors);
          continue;
        }
      local = std::make_unique<LocalPixel>(localCR);  // local pixel coordinate
      //if(DANEK) cout<<local->dcol()<<" "<<local->pxid()<<" "<<local->rocCol()<<" "<<local->rocRow()<<endl;

    } else {  // phase0 and phase1 except bpix layer 1
      int dcol = (ww >> DCOL_shift) & DCOL_mask;
      int pxid = (ww >> PXID_shift) & PXID_mask;
      //if(DANEK) cout<<" raw2digi "<<link<<" "<<roc<<" "
      //<<dcol<<" "<<pxid<<" "<<adc<<" "<<layer<<endl;

      LocalPixel::DcolPxid localDP = {dcol, pxid};
      //if(DANEK) cout<<localDP.dcol<<" "<<localDP.pxid<<endl;

      if
        UNLIKELY(!localDP.valid()) {
          LogDebug("PixelDataFormatter::interpretRawData") << "status #3";
          errorsInEvent = true;
          errorcheck->conversionError(fedId, &converter, 3, ww, errors);
          continue;
        }
      local = std::make_unique<LocalPixel>(localDP);  // local pixel coordinate
      //if(DANEK) cout<<local->dcol()<<" "<<local->pxid()<<" "<<local->rocCol()<<" "<<local->rocRow()<<endl;
    }

    GlobalPixel global = rocp->toGlobal(*local);  // global pixel coordinate (in module)
    (*detDigis).data.emplace_back(global.row, global.col, adc);
    //if(DANEK) cout<<global.row<<" "<<global.col<<" "<<adc<<endl;
    LogTrace("") << (*detDigis).data.back();
  }
}

// I do not know what this was for or if it is needed? d.k. 10.14
// Keep it commented out until we are sure that it is not needed.
// void doVectorize(int const * __restrict__ w, int * __restrict__ row, int * __restrict__ col, int * __restrict__ valid, int N, PixelROC const * rocp) {
//   for (int i=0; i<N; ++i) {
//     auto ww = w[i];
//     int dcol = (ww >> DCOL_shift) & DCOL_mask;
//     int pxid = (ww >> PXID_shift) & PXID_mask;
//     // int adc  = (ww >> ADC_shift) & ADC_mask;
//     LocalPixel::DcolPxid local = { dcol, pxid };
//     valid[i] = local.valid();
//     GlobalPixel global = rocp->toGlobal( LocalPixel(local) );
//     row[i]=global.row; col[i]=global.col;
//   }
// }

void PixelDataFormatter::formatRawData(unsigned int lvl1_ID,
                                       RawData& fedRawData,
                                       const Digis& digis,
                                       const BadChannels& badChannels) {
  std::map<int, vector<Word32> > words;

  // translate digis into 32-bit raw words and store in map indexed by Fed
  for (Digis::const_iterator im = digis.begin(); im != digis.end(); im++) {
    allDetDigis++;
    cms_uint32_t rawId = im->first;
    int layer = 0;
    bool barrel = PixelModuleName::isBarrel(rawId);
    if (barrel)
      layer = PixelROC::bpixLayerPhase1(rawId);
    //if(DANEK) cout<<" layer "<<layer<<" "<<phase1<<endl;

    BadChannels::const_iterator detBadChannels = badChannels.find(rawId);

    hasDetDigis++;
    const DetDigis& detDigis = im->second;
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); it++) {
      theDigiCounter++;
      const PixelDigi& digi = (*it);
      int fedId = 0;

      if (layer == 1 && phase1)
        fedId = digi2wordPhase1Layer1(rawId, digi, words);
      else
        fedId = digi2word(rawId, digi, words);

      if (fedId < 0) {
        LogError("FormatDataException") << " digi2word returns error #" << fedId << " Ndigis: " << theDigiCounter
                                        << endl
                                        << " detector: " << rawId << endl
                                        << print(digi) << endl;
      } else if (detBadChannels != badChannels.end()) {
        auto badChannel =
            std::find_if(detBadChannels->second.begin(), detBadChannels->second.end(), [&](const PixelFEDChannel& ch) {
              return (int(ch.fed) == fedId && ch.link == linkId(words[fedId].back()));
            });
        if (badChannel != detBadChannels->second.end()) {
          LogError("FormatDataException") << " while marked bad, found digi for FED " << fedId << " Link "
                                          << linkId(words[fedId].back()) << " on module " << rawId << endl
                                          << print(digi) << endl;
        }
      }  // if (fedId)
    }    // for (DetDigis
  }      // for (Digis
  LogTrace(" allDetDigis/hasDetDigis : ") << allDetDigis << "/" << hasDetDigis;

  // fill FED error 25 words
  for (const auto& detBadChannels : badChannels) {
    for (const auto& badChannel : detBadChannels.second) {
      unsigned int FEDError25 = 25;
      Word32 word = (badChannel.link << LINK_shift) | (FEDError25 << ROC_shift);
      words[badChannel.fed].push_back(word);
      theWordCounter++;
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
  LogDebug("PixelDataFormatter")
      // <<" detId: " << detId
      << print(digi);

  DetectorIndex detector = {detId, digi.row(), digi.column()};
  ElectronicIndex cabling;
  int fedId = theFrameReverter->toCabling(cabling, detector);
  if (fedId < 0)
    return fedId;

  //if(DANEK) cout<<" digi2raw "<<detId<<" "<<digi.column()<<" "<<digi.row()<<" "<<digi.adc()<<" "
  //  <<cabling.link<<" "<<cabling.roc<<" "<<cabling.dcol<<" "<<cabling.pxid<<endl;

  Word32 word = (cabling.link << LINK_shift) | (cabling.roc << ROC_shift) | (cabling.dcol << DCOL_shift) |
                (cabling.pxid << PXID_shift) | (digi.adc() << ADC_shift);
  words[fedId].push_back(word);
  theWordCounter++;

  return fedId;
}
int PixelDataFormatter::digi2wordPhase1Layer1(cms_uint32_t detId,
                                              const PixelDigi& digi,
                                              std::map<int, vector<Word32> >& words) const {
  LogDebug("PixelDataFormatter")
      // <<" detId: " << detId
      << print(digi);

  DetectorIndex detector = {detId, digi.row(), digi.column()};
  ElectronicIndex cabling;
  int fedId = theFrameReverter->toCabling(cabling, detector);
  if (fedId < 0)
    return fedId;

  int col = ((cabling.dcol) * 2) + ((cabling.pxid) % 2);
  int row = LocalPixel::numRowsInRoc - ((cabling.pxid) / 2);

  //if(DANEK) cout<<" layer 1: digi2raw "<<detId<<" "<<digi.column()<<" "<<digi.row()<<" "<<digi.adc()<<" "
  //  <<cabling.link<<" "<<cabling.roc<<" "<<cabling.dcol<<" "<<cabling.pxid<<" "
  //  <<col<<" "<<row<<endl;

  Word32 word = (cabling.link << LINK_shift) | (cabling.roc << ROC_shift) | (col << COL_shift) | (row << ROW_shift) |
                (digi.adc() << ADC_shift);
  words[fedId].push_back(word);
  theWordCounter++;

  return fedId;
}

// obsolete...
int PixelDataFormatter::word2digi(const int fedId,
                                  const SiPixelFrameConverter* converter,
                                  const bool includeErrors,
                                  const bool useQuality,
                                  const Word32& word,
                                  Digis& digis) const {
  // do not interpret false digis
  if (word == 0)
    return 0;

  ElectronicIndex cabling;
  cabling.dcol = (word >> DCOL_shift) & DCOL_mask;
  cabling.pxid = (word >> PXID_shift) & PXID_mask;
  cabling.link = (word >> LINK_shift) & LINK_mask;
  cabling.roc = (word >> ROC_shift) & ROC_mask;
  int adc = (word >> ADC_shift) & ADC_mask;

  if (debug) {
    LocalPixel::DcolPxid pixel = {cabling.dcol, cabling.pxid};
    LocalPixel local(pixel);
    LogTrace("") << " link: " << cabling.link << ", roc: " << cabling.roc << " rocRow: " << local.rocRow()
                 << ", rocCol:" << local.rocCol() << " (dcol: " << cabling.dcol << ", pxid:" << cabling.pxid
                 << "), adc:" << adc;
  }

  if (!converter)
    return 0;

  DetectorIndex detIdx;
  int status = converter->toDetector(cabling, detIdx);
  if (status)
    return status;

  // exclude ROC(raw) based on bad ROC list bad in SiPixelQuality
  // enable: process.siPixelDigis.UseQualityInfo = True
  // 20-10-2010 A.Y.
  if (useQuality && badPixelInfo) {
    CablingPathToDetUnit path = {static_cast<unsigned int>(fedId),
                                 static_cast<unsigned int>(cabling.link),
                                 static_cast<unsigned int>(cabling.roc)};
    const PixelROC* roc = theCablingTree->findItem(path);
    short rocInDet = (short)roc->idInDetUnit();
    bool badROC = badPixelInfo->IsRocBad(detIdx.rawId, rocInDet);
    if (badROC)
      return 0;
  }

  if (modulesToUnpack && modulesToUnpack->find(detIdx.rawId) == modulesToUnpack->end())
    return 0;

  digis[detIdx.rawId].emplace_back(detIdx.row, detIdx.col, adc);

  theDigiCounter++;

  if (debug)
    LogTrace("") << digis[detIdx.rawId].back();
  return 0;
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
