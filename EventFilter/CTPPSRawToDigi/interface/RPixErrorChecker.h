#ifndef CTPPS_CTPPSRawToDigi_RPixErrorChecker_h
#define CTPPS_CTPPSRawToDigi_RPixErrorChecker_h
/** \class RPixErrorChecker
 *
 *  RPix == CTPPS Pixel detector (Roman Pot Pixels)
 */

#include <cstdint>

#include <vector>
#include <map>

class FEDRawData;

class RPixErrorChecker {

public:

  typedef uint32_t Word32;
  typedef uint64_t Word64;



  RPixErrorChecker();

  bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer) const;

  bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header) const;

  bool checkTrailer(bool& errorsInEvent, int fedId, int nWords, const Word64* trailer) const;

  bool checkROC(bool& errorsInEvent, int fedId, Word32& errorWord) const;


private:


};

#endif
