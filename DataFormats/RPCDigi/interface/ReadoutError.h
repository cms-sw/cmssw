#ifndef DataFormats_RPCDigi_ReadoutError_H
#define DataFormats_RPCDigi_ReadoutError_H

#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
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

    explicit ReadoutError(unsigned int rawCode = 0) : theError(rawCode) {}

    ReadoutError(const LinkBoardElectronicIndex&, const ReadoutErrorType&);

    ReadoutErrorType type() const;
    LinkBoardElectronicIndex where() const;

    static std::string name(const ReadoutErrorType& code);

    std::string name() const { return name(type()); }

    unsigned int rawCode() const { return theError; }

  private:
    unsigned int theError;
  };

}  // namespace rpcrawtodigi
#endif
