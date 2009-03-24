#ifndef DataFormats_RPCDigi_ReadoutError_H
#define DataFormats_RPCDigi_ReadoutError_H

#include <string>

namespace rpcrawtodigi { 
class ReadoutError {
public:

  enum ReadoutErrorType { 
      NoProblem = 0,
      HeaderCheckFail = 1,
      InconsitentFedId = 2,
      TrailerCheckFail = 3,
      InconsistentDataSize = 4,
      InvalidLB = 5,
      EmptyPackedStrips = 6,
      InvalidDetId = 7,
      InvalidStrip = 8, 
      EOD = 9 
  };

  explicit ReadoutError(const ReadoutErrorType & type = NoProblem) : theType(type) { }

  ReadoutErrorType type() const { return theType; }

  static std::string name(const ReadoutErrorType & code);

  std::string name() const { return name(theType); }

private:
  ReadoutErrorType theType;
};

}
#endif

