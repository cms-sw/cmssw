#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

//---------------------------------------------------------------------------
//!  \class SiPixelRawDataError
//!  \brief Pixel error -- collection of errors and error information
//!
//!  Class to contain and store all information about errors
//!
//!
//!  \author Andrew York, University of Tennessee
//---------------------------------------------------------------------------

//Constructors

SiPixelRawDataError::SiPixelRawDataError() {}

SiPixelRawDataError::SiPixelRawDataError(cms_uint32_t errorWord32, const int errorType, int fedId)
    : errorWord64_(0), errorWord32_(errorWord32), errorType_(errorType), fedId_(fedId) {}

SiPixelRawDataError::SiPixelRawDataError(cms_uint64_t errorWord64, const int errorType, int fedId)
    : errorWord64_(errorWord64), errorWord32_(0), errorType_(errorType), fedId_(fedId) {}

//Destructor

SiPixelRawDataError::~SiPixelRawDataError() {}

//functions to get error words and types

void SiPixelRawDataError::setWord32(cms_uint32_t errorWord32) { errorWord32_ = errorWord32; }

void SiPixelRawDataError::setWord64(cms_uint64_t errorWord64) { errorWord64_ = errorWord64; }

void SiPixelRawDataError::setType(int errorType) { errorType_ = errorType; }

void SiPixelRawDataError::setFedId(int fedId) { fedId_ = fedId; }

std::string_view SiPixelRawDataError::getMessage() const {
  switch (errorType_) {
    case (25): {
      return "Error: Disabled FED channel (ROC=25)";
      break;
    }
    case (26): {
      return "Error: Gap word";
      break;
    }
    case (27): {
      return "Error: Dummy word";
      break;
    }
    case (28): {
      return "Error: FIFO nearly full";
      break;
    }
    case (29): {
      return "Error: Timeout";
      break;
    }
    case (30): {
      return "Error: Trailer";
      break;
    }
    case (31): {
      return "Error: Event number mismatch";
      break;
    }
    case (32): {
      return "Error: Invalid or missing header";
      break;
    }
    case (33): {
      return "Error: Invalid or missing trailer";
      break;
    }
    case (34): {
      return "Error: Size mismatch";
      break;
    }
    case (35): {
      return "Error: Invalid channel";
      break;
    }
    case (36): {
      return "Error: Invalid ROC number";
      break;
    }
    case (37): {
      return "Error: Invalid dcol/pixel address";
      break;
    }
    default:
      return "Error: Unknown error type";
  };
}
