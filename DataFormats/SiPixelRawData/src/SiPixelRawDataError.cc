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

SiPixelRawDataError::SiPixelRawDataError(const unsigned int errorWord32, const int errorType) : 
  errorWord32_(errorWord32),
  errorType_(errorType),
  detId_(0xffffffff)
  {
    setMessage(errorType);
  }

SiPixelRawDataError::SiPixelRawDataError(const unsigned int errorWord32, const int errorType, uint32_t detId) : 
  errorWord32_(errorWord32),
  errorType_(errorType),
  detId_(detId)
  {
    setMessage(errorType);
  }

SiPixelRawDataError::SiPixelRawDataError(const long long errorWord64, const int errorType) : 
  errorWord64_(errorWord64),
  errorType_(errorType),
  detId_(0xffffffff)
  {
    setMessage(errorType);
  }

//Destructor

SiPixelRawDataError::~SiPixelRawDataError() {}

//functions to get error words and types

void SiPixelRawDataError::setWord32(unsigned int errorWord32) {
  errorWord32_ = errorWord32;
}

void SiPixelRawDataError::setWord64(long long errorWord64) {
  errorWord64_ = errorWord64;
}

void SiPixelRawDataError::setType(int errorType) {
errorType_ = errorType; 
setMessage(errorType);
} 

void SiPixelRawDataError::setDetId(uint32_t detId) {
detId_ = detId; 
} 

void SiPixelRawDataError::setMessage(int errorType) {
  switch (errorType) {
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
