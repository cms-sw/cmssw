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

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include <boost/cstdint.hpp>
#include <vector>
#include <map>

class FEDRawData;

class SiPixelFedCablingMap;
class SiPixelFrameConverter;

class PixelDataFormatter {

public:

  typedef std::vector<PixelDigi> DetDigis;
  typedef std::map<uint32_t, DetDigis> Digis;
//  typedef std::vector< edm::DetSet<PixelDigi> > Digis;
  typedef std::pair<DetDigis::const_iterator, DetDigis::const_iterator> Range;
  
  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<uint32_t, DetErrors> Errors;

  PixelDataFormatter(const SiPixelFedCablingMap * map);

  int nDigis() const { return theDigiCounter; }
  int nWords() const { return theWordCounter; }

  void interpretRawData(int fedId,  const FEDRawData & data, Digis & digis, bool includeErrors, Errors & errors);

  FEDRawData * formatData( int fedId, const Digis & digis);

private:
  mutable int theDigiCounter;
  mutable int theWordCounter;

  const SiPixelFedCablingMap * theCablingMap;

  typedef unsigned int Word32;
  typedef long long Word64;
//  typedef uint32_t Word32;
//  typedef uint64_t Word64;

  int checkError(const Word32& data) const;

  int digi2word( const SiPixelFrameConverter* converter,
                  uint32_t detId, const PixelDigi& digi,
                  std::vector<Word32> & words) const;

  int word2digi( const SiPixelFrameConverter* converter, 
		    const bool includeError,
		    const Word32& word, 
                    Digis & digis) const;

  std::string print(const PixelDigi & digi) const;
  std::string print(const Word64    & word) const;

  uint32_t errorDetId(const SiPixelFrameConverter* converter, 
		      const Word32 & word) const;

  static const int LINK_bits,  ROC_bits,  DCOL_bits,  PXID_bits,  ADC_bits;
  static const int LINK_shift, ROC_shift, DCOL_shift, PXID_shift, ADC_shift;
};

#endif
