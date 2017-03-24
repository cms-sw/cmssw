#ifndef RPixErrorChecker_H
#define RPixErrorChecker_H
/** \class RPixErrorChecker
 *
 *  RPix == CTPPS Pixel detector (Roman Pot Pixels)
 */

#include "FWCore/Utilities/interface/typedefs.h"

#include <vector>
#include <map>

class FEDRawData;

class RPixErrorChecker {

public:

  typedef cms_uint32_t Word32;
  typedef cms_uint64_t Word64;



  RPixErrorChecker();

  bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer);

  bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header);

  bool checkTrailer(bool& errorsInEvent, int fedId, int nWords, const Word64* trailer);

  bool checkROC(bool& errorsInEvent, int fedId, Word32& errorWord);


private:


};

#endif
