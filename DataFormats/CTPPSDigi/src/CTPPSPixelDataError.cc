
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDataError.h"

const std::string CTPPSPixelDataError::errorMessages[] = {

    "Error: Unknown error type", 
      // 25) : {
      "Error: ROC=25",
      // 26) : {
      "Error: Gap word",
      // 27) : {
      "Error: Dummy word",
      // 28) : {
      "Error: FIFO nearly full",
      // 29) : {
      "Error: Timeout",
      // 30) : {
      "Error: Trailer",
      // 31) : {
      "Error: Event number mismatch",
      // 32) : {
      "Error: Invalid or missing header",
      // 33) : {
      "Error: Invalid or missing trailer",
      // 34) : {
      "Error: Size mismatch",
      // 35) : {
      "Error: Invalid channel",
      // 36) : {
      "Error: Invalid ROC number",
      // 37) : {
      "Error: Invalid dcol/pixel address"
}; 

CTPPSPixelDataError::CTPPSPixelDataError():
  errorWord64_(0),
  errorWord32_(0),
  errorType_(0),
  fedId_(0) 
{}

CTPPSPixelDataError::CTPPSPixelDataError(uint32_t errorWord32, const int errorType, int fedId) : 
  errorWord64_(0),
  errorWord32_(errorWord32),
  errorType_(errorType),
  fedId_(fedId)
{
  setMessage();
}

CTPPSPixelDataError::CTPPSPixelDataError(uint64_t errorWord64, const int errorType, int fedId) : 
  errorWord64_(errorWord64),
  errorWord32_(0),
  errorType_(errorType),
  fedId_(fedId)
{
  setMessage();
}

CTPPSPixelDataError::~CTPPSPixelDataError() = default;

void CTPPSPixelDataError::setMessage() {
  if(errorType_<25 || errorType_ > 37){
    errorMessage_ = errorMessages[0];
  }else{
    errorMessage_ = errorMessages[errorType_-24];
  }
}

