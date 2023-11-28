#ifndef DataFormats_CTPPSPixelDataError_h
#define DataFormats_CTPPSPixelDataError_h

//---------------------------------------------------------------------------
//!  \class CTPPSPixelDataError
//!  \brief Pixel error -- collection of errors
//!
//!  Class storing info about pixel errors
//!
//---------------------------------------------------------------------------

#include "FWCore/Utilities/interface/typedefs.h"

#include <string>
#include <array>
#include <cstdint>

class CTPPSPixelDataError {
public:
  /// Default constructor
  CTPPSPixelDataError();
  /// Constructor for 32-bit error word
  CTPPSPixelDataError(uint32_t errorWord32, const int errorType, int fedId);
  /// Constructor with 64-bit error word and type included (header or trailer word)
  CTPPSPixelDataError(uint64_t errorWord64, const int errorType, int fedId);
  /// Destructor
  ~CTPPSPixelDataError();

  /// the 32-bit word that contains the error information
  inline uint32_t errorWord32() const { return errorWord32_; }
  /// the 64-bit word that contains the error information
  inline uint64_t errorWord64() const { return errorWord64_; }
  /// the number associated with the error type (26-31 for ROC number errors, 32-33 for calibration errors)
  inline int errorType() const { return errorType_; }
  /// the fedId where the error occured
  inline int fedId() const { return fedId_; }
  /// the error message to be displayed with the error
  inline const std::string& errorMessage() const {
    if (errorType_ >= 25 && errorType_ <= 37)
      return errorMessages_[errorType_ - 24];
    return errorMessages_[0];
  }

private:
  static const std::array<std::string, 14> errorMessages_;

  uint64_t errorWord64_;
  uint32_t errorWord32_;
  int errorType_;
  int fedId_;
};

/// Comparison operators
inline bool operator<(const CTPPSPixelDataError& one, const CTPPSPixelDataError& other) {
  return one.fedId() < other.fedId();
}

#endif
