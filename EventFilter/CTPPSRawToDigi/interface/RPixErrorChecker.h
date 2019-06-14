#ifndef CTPPS_CTPPSRawToDigi_RPixErrorChecker_h
#define CTPPS_CTPPSRawToDigi_RPixErrorChecker_h
/** \class RPixErrorChecker
 *
 *  RPix == CTPPS Pixel detector (Roman Pot Pixels)
 */

#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDataError.h"

#include <vector>
#include <map>

class FEDRawData;

enum State { InvalidLinkId, InvalidROCId, InvalidPixelId, Unknown };

class RPixErrorChecker {
public:
  typedef uint32_t Word32;
  typedef uint64_t Word64;

  typedef std::vector<CTPPSPixelDataError> DetErrors;
  typedef std::map<uint32_t, DetErrors> Errors;

  static constexpr int CRC_bits = 1;
  static constexpr int ROC_bits = 5;
  static constexpr int DCOL_bits = 5;
  static constexpr int PXID_bits = 8;
  static constexpr int ADC_bits = 8;
  static constexpr int OMIT_ERR_bits = 1;

  static constexpr int CRC_shift = 2;
  static constexpr int ADC_shift = 0;
  static constexpr int PXID_shift = ADC_shift + ADC_bits;
  static constexpr int DCOL_shift = PXID_shift + PXID_bits;
  static constexpr int ROC_shift = DCOL_shift + DCOL_bits;
  static constexpr int OMIT_ERR_shift = 20;

  static constexpr Word32 dummyDetId = 0xffffffff;

  static constexpr Word64 CRC_mask = ~(~RPixErrorChecker::Word64(0) << CRC_bits);
  static constexpr Word32 ERROR_mask = ~(~RPixErrorChecker::Word32(0) << ROC_bits);
  static constexpr Word32 OMIT_ERR_mask = ~(~RPixErrorChecker::Word32(0) << OMIT_ERR_bits);

public:
  RPixErrorChecker();

  void setErrorStatus(bool errorStatus);

  bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors) const;

  bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors) const;

  bool checkTrailer(bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors) const;

  bool checkROC(bool& errorsInEvent, int fedId, uint32_t iD, const Word32& errorWord, Errors& errors) const;

  void conversionError(int fedId, uint32_t iD, const State& state, const Word32& errorWord, Errors& errors) const;

private:
  bool includeErrors_;
};

#endif
