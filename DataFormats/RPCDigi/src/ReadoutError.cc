#include "DataFormats/RPCDigi/interface/ReadoutError.h"
#include <bitset>
#include <iostream>

using namespace rpcrawtodigi;

ReadoutError::ReadoutError(const LinkBoardElectronicIndex &path, const ReadoutErrorType &type) {
  unsigned int where =
      (path.dccId << 13) | (path.dccInputChannelNum << 7) | (path.tbLinkInputNum << 2) | path.lbNumInLink;
  unsigned int what = type;
  theError = (where << 4) | (what & 0xf);
}

ReadoutError::ReadoutErrorType ReadoutError::type() const {
  //return static_cast<ReadoutErrorType>(theError&0xf);
  return ReadoutErrorType(theError & 0xf);
}

LinkBoardElectronicIndex ReadoutError::where() const {
  unsigned int data = (theError >> 4);
  LinkBoardElectronicIndex ele;
  ele.dccId = (data >> 13);
  ele.dccInputChannelNum = (data >> 7) & 63;
  ele.tbLinkInputNum = (data >> 2) & 31;
  ele.lbNumInLink = data & 3;
  return ele;
}

std::string ReadoutError::name(const ReadoutErrorType &code) {
  std::string result;
  switch (code) {
    case (HeaderCheckFail): {
      result = "HeaderCheckFail";
      break;
    }
    case (InconsitentFedId): {
      result = "InconsitentFedId";
      break;
    }
    case (TrailerCheckFail): {
      result = "TrailerCheckFail";
      break;
    }
    case (InconsistentDataSize): {
      result = "InconsistentDataSize";
      break;
    }
    case (InvalidLB): {
      result = "InvalidLB";
      break;
    }
    case (EmptyPackedStrips): {
      result = "EmptyPackedStrips";
      break;
    }
    case (InvalidDetId): {
      result = "InvalidDetId";
      break;
    }
    case (InvalidStrip): {
      result = "InvalidStrip";
      break;
    }
    case (EOD): {
      result = "EOD";
      break;
    }
    default: {
      result = "NoProblem";
    }
  }
  return result;
}
