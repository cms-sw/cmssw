
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDataError.h"

const std::array<std::string, 14> CTPPSPixelDataError::errorMessages_ = {{

    "Error: Unknown error type",
    /// error 25
    "Error: ROC=25",
    /// error  26
    "Error: Gap word",
    /// error  27
    "Error: Dummy word",
    /// error  28
    "Error: FIFO nearly full",
    /// error  29
    "Error: Timeout",
    /// error  30
    "Error: Trailer",
    /// error  31
    "Error: Event number mismatch",
    /// error  32
    "Error: Invalid or missing header",
    /// error  33
    "Error: Invalid or missing trailer",
    /// error  34
    "Error: Size mismatch",
    /// error  35
    "Error: Invalid channel",
    /// error  36
    "Error: Invalid ROC number",
    /// error  37
    "Error: Invalid dcol/pixel address"}};

CTPPSPixelDataError::CTPPSPixelDataError() : errorWord64_(0), errorWord32_(0), errorType_(0), fedId_(0) {}

CTPPSPixelDataError::CTPPSPixelDataError(uint32_t errorWord32, const int errorType, int fedId)
    : errorWord64_(0), errorWord32_(errorWord32), errorType_(errorType), fedId_(fedId) {}

CTPPSPixelDataError::CTPPSPixelDataError(uint64_t errorWord64, const int errorType, int fedId)
    : errorWord64_(errorWord64), errorWord32_(0), errorType_(errorType), fedId_(fedId) {}

CTPPSPixelDataError::~CTPPSPixelDataError() = default;
