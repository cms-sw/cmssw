#include "DataFormats/RPCDigi/interface/ReadoutError.h"

using namespace rpcrawtodigi;

std::string ReadoutError::name(const ReadoutErrorType & code) 
{
  std::string result;
  switch (code) {
    case (HeaderCheckFail)      : { result = "HeaderCheckFail"; break; }
    case (InconsitentFedId)     : { result = "InconsitentFedId"; break; }
    case (TrailerCheckFail)     : { result = "TrailerCheckFail"; break; }
    case (InconsistentDataSize) : { result = "InconsistentDataSize"; break; }
    case (InvalidLB)            : { result = "InvalidLB"; break; }
    case (EmptyPackedStrips)    : { result = "EmptyPackedStrips"; break; }
    case (InvalidDetId)         : { result = "InvalidDetId"; break; }
    case (InvalidStrip)         : { result = "InvalidStrip"; break; }
    case (EOD)                  : { result = "EOD"; break; }
    default                     : { result = "NoProblem"; }
  }
  return result;
}

