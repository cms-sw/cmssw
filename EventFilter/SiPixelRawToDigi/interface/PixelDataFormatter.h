#ifndef EventFilter_SiPixelRawToDigi_interface_PixelDataFormatter_h
#define EventFilter_SiPixelRawToDigi_interface_PixelDataFormatter_h
/** \class PixelDataFormatter
 *
 *  Transforms Pixel raw data of a given  FED to orca digi
 *  and vice versa.
 *
 * FED OUTPUT DATA FORMAT 6/02, d.k.  (11/02 updated for 100*150 pixels)
 * ----------------------
 * The output is transmitted through a 64 bit S-link connection.
 * The packet format is defined by the CMS RU group to be :
 * 1st packet header, 64 bits, includes a 6 bit FED id.
 * 2nd packet header, 64 bits.
 * .......................... (detector data)
 * packet trailer, 64 bits.
 * of the 64 bit pixel data records consists of 2
 * 32 bit words. Each 32 bit word includes data from 1 pixel,
 * the bit fields are the following:
 *
 * 6 bit link ID (max 36)   - this defines the input link within 1 FED.
 * 5 bit ROC ID (max 24)    - this defines the readout chip within one link.
 * 5 bit DCOL ID (max 26)   - this defines the double column index with 1 chip.
 * 8 bit pixel ID (max 180) - this defines the pixel address within 1 DCOL.
 * 8 bit ADC vales          - this has the charge amplitude.
 *
 * So, 1 pixel occupies 4 bytes.
 * If the number of pixels is odd, one extra 32 bit word is added (value 0)
 * to fill all 64 bits.
 *
 * The PixelDataFormatter interpret/format ONLY detector data words
 * (not FED headers or trailer, which are treated elsewhere).
 */
//
// Add the phase1 format
//
// CMSSW include(s)
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/SiPixelRawToDigi/interface/ErrorChecker.h"
#include "EventFilter/SiPixelRawToDigi/interface/ErrorCheckerPhase0.h"
#include "FWCore/Utilities/interface/typedefs.h"

// standard include(s)
#include <vector>
#include <map>
#include <set>

class FEDRawData;
class SiPixelFedCabling;
class SiPixelQuality;
class SiPixelFrameConverter;
class SiPixelFrameReverter;
class SiPixelFedCablingTree;

class PixelDataFormatter {
public:
  using DetErrors = std::vector<SiPixelRawDataError>;
  using Errors = std::map<cms_uint32_t, DetErrors>;
  using Collection = edm::DetSetVector<PixelDigi>;
  using RawData = std::map<int, FEDRawData>;
  using DetDigis = std::vector<PixelDigi>;
  using Digis = std::map<cms_uint32_t, DetDigis>;
  using DetBadChannels = std::vector<PixelFEDChannel>;
  using BadChannels = std::map<cms_uint32_t, DetBadChannels>;
  using FEDWordsMap = std::map<int, std::vector<Word32>>;
  using ModuleIDSet = std::set<unsigned int>;

  PixelDataFormatter(const SiPixelFedCablingTree* map, bool phase1_ = false);

  void setErrorStatus(bool ErrorStatus);
  void setQualityStatus(bool QualityStatus, const SiPixelQuality* QualityInfo);
  void setModulesToUnpack(const ModuleIDSet* moduleIds);
  void passFrameReverter(const SiPixelFrameReverter* reverter);

  int nDigis() const { return theDigiCounter_; }
  int nWords() const { return theWordCounter_; }

  void interpretRawData(bool& errorsInEvent, int fedId, const FEDRawData& data, Collection& digis, Errors& errors);

  void formatRawData(unsigned int lvl1_ID, RawData& fedRawData, const Digis& digis, const BadChannels& badChannels);

  void unpackFEDErrors(Errors const& errors,
                       std::vector<int> const& tkerrorlist,
                       std::vector<int> const& usererrorlist,
                       edm::DetSetVector<SiPixelRawDataError>& errorcollection,
                       DetIdCollection& tkerror_detidcollection,
                       DetIdCollection& usererror_detidcollection,
                       edmNew::DetSetVector<PixelFEDChannel>& disabled_channelcollection,
                       DetErrors& nodeterrors);

private:
  mutable int theDigiCounter_;
  mutable int theWordCounter_;

  SiPixelFedCablingTree const* theCablingTree_;
  const SiPixelFrameReverter* theFrameReverter_;
  const SiPixelQuality* badPixelInfo_;
  const ModuleIDSet* modulesToUnpack_;

  bool includeErrors_;
  bool useQualityInfo_;
  int allDetDigis_;
  int hasDetDigis_;
  std::unique_ptr<ErrorCheckerBase> errorcheck_;

  int maxROCIndex_;
  bool phase1_;

  int checkError(const Word32& data) const;

  int digi2word(cms_uint32_t detId, const PixelDigi& digi, FEDWordsMap& words) const;
  int digi2wordPhase1Layer1(cms_uint32_t detId, const PixelDigi& digi, FEDWordsMap& words) const;

  std::string print(const PixelDigi& digi) const;
  std::string print(const Word64& word) const;

  cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, int fedId, int errorType, const Word32& word) const;
};

#endif  // EventFilter_SiPixelRawToDigi_interface_PixelDataFormatter_h
