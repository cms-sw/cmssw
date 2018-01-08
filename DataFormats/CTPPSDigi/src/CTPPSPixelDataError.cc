
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDataError.h"


CTPPSPixelDataError::CTPPSPixelDataError() {}

CTPPSPixelDataError::CTPPSPixelDataError(uint32_t errorWord32, const int errorType, int fedId) : 
  errorWord32_(errorWord32),
  errorType_(errorType),
  fedId_(fedId)
{
  setMessage();
}

CTPPSPixelDataError::CTPPSPixelDataError(uint64_t errorWord64, const int errorType, int fedId) : 
  errorWord64_(errorWord64),
  errorType_(errorType),
  fedId_(fedId)
{
  setMessage();
}

CTPPSPixelDataError::~CTPPSPixelDataError() = default;

void CTPPSPixelDataError::setWord32(uint32_t errorWord32) {
  errorWord32_ = errorWord32;
}

void CTPPSPixelDataError::setWord64(uint64_t errorWord64) {
  errorWord64_ = errorWord64;
}

void CTPPSPixelDataError::setType(int errorType) {
  errorType_ = errorType; 
  setMessage();
} 

void CTPPSPixelDataError::setFedId(int fedId) {
  fedId_ = fedId; 
} 

void CTPPSPixelDataError::setMessage() {
  switch (errorType_) {
  case(25) : {
    errorMessage_ = "Error: ROC=25";
    break;
  }
  case(26) : {
    errorMessage_ = "Error: Gap word";
    break;
  }
  case(27) : {
    errorMessage_ = "Error: Dummy word";
    break;
  }
  case(28) : {
    errorMessage_ = "Error: FIFO nearly full";
    break;
  }
  case(29) : {
    errorMessage_ = "Error: Timeout";
    break;
  }
  case(30) : {
    errorMessage_ = "Error: Trailer";
    break;
  }
  case(31) : {
    errorMessage_ = "Error: Event number mismatch";
    break;
  }
  case(32) : {
    errorMessage_ = "Error: Invalid or missing header";
    break;
  }
  case(33) : {
    errorMessage_ = "Error: Invalid or missing trailer";
    break;
  }
  case(34) : {
    errorMessage_ = "Error: Size mismatch";
    break;
  }
  case(35) : {
    errorMessage_ = "Error: Invalid channel";
    break;
  }
  case(36) : {
    errorMessage_ = "Error: Invalid ROC number";
    break;
  }
  case(37) : {
    errorMessage_ = "Error: Invalid dcol/pixel address";
    break;
  }
  default: errorMessage_ = "Error: Unknown error type";
  };
} 
