#ifndef PixelDataFormatter_H
#define PixelDataFormatter_H
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

#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "EventFilter/SiPixelRawToDigi/interface/ErrorChecker.h"
#include "FWCore/Utilities/interface/typedefs.h"

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

  typedef edm::DetSetVector<PixelDigi> Collection;

  typedef std::map<int, FEDRawData> RawData;
  typedef std::vector<PixelDigi> DetDigis;
  typedef std::map<cms_uint32_t, DetDigis> Digis;
  typedef std::pair<DetDigis::const_iterator, DetDigis::const_iterator> Range; 
  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<cms_uint32_t, DetErrors> Errors;

  typedef cms_uint32_t Word32;
  typedef cms_uint64_t Word64;

  PixelDataFormatter(const SiPixelFedCabling* map);

  void setErrorStatus(bool ErrorStatus);
  void setQualityStatus(bool QualityStatus, const SiPixelQuality* QualityInfo);
  void setModulesToUnpack(const std::set<unsigned int> * moduleIds);
  void passFrameReverter(const SiPixelFrameReverter* reverter);

  int nDigis() const { return theDigiCounter; }
  int nWords() const { return theWordCounter; }

  void interpretRawData(bool& errorsInEvent, int fedId,  const FEDRawData & data, Collection & digis, Errors & errors);

  void formatRawData( unsigned int lvl1_ID, RawData & fedRawData, const Digis & digis);

private:
  mutable int theDigiCounter;
  mutable int theWordCounter;


  SiPixelFedCabling const * theCablingTree;
  const SiPixelFrameReverter* theFrameReverter;
  const SiPixelQuality* badPixelInfo;
  const std::set<unsigned int> * modulesToUnpack;

  bool includeErrors;
  bool useQualityInfo;
  bool debug;
  int allDetDigis;
  int hasDetDigis;
  ErrorChecker errorcheck;

  int checkError(const Word32& data) const;

  int digi2word(  cms_uint32_t detId, const PixelDigi& digi,
                  std::map<int, std::vector<Word32> > & words) const;

  int word2digi(  const int fedId,
                  const SiPixelFrameConverter* converter,
		  const bool includeError,
		  const bool useQuality,
		  const Word32& word, 
                  Digis & digis) const;

  std::string print(const PixelDigi & digi) const;
  std::string print(const Word64    & word) const;

  cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, 
		      int fedId, int errorType, const Word32 & word) const;

};

#endif
