//---------------------------------------------------

// Author : Freya.Blekman@NOSPAM.cern.ch
// Name   : SiPixelCalibDigiError

//---------------------------------------------------

#ifndef DataFormats_SiPixelCalibDigi_SiPixelCalibDigiError_H
#define DataFormats_SiPixelCalibDigi_SiPixelCalibDigiError_H

#include <iostream>
#include <utility>
#include <string>
#include <cstdint>

class SiPixelCalibDigiError {
private:
  uint16_t fRow;       // row number
  uint16_t fCol;       // column number
  uint8_t fErrorType;  // error type in case we want to label errors

public:
  SiPixelCalibDigiError() : fRow(0), fCol(0), fErrorType(0) { ; }

  SiPixelCalibDigiError(uint16_t row, uint16_t col) : fRow(row), fCol(col), fErrorType(0) { ; }
  SiPixelCalibDigiError(uint16_t row, uint16_t col, uint16_t error) : fRow(row), fCol(col), fErrorType(error) { ; }

  virtual ~SiPixelCalibDigiError() { ; }

  //- Accessible methods

  //-- Setter/Getter

  uint16_t getRow() const { return fRow; }
  void setRow(const uint16_t& in) { fRow = in; }

  uint16_t getCol() const { return fCol; }
  void setCol(const uint16_t& in) { fCol = in; }

  uint16_t getErrorType() const { return fErrorType; }
  void setErrorType(const uint16_t& in) { fErrorType = in; }
  std::string printError() const;
};
inline bool operator<(const SiPixelCalibDigiError& one, const SiPixelCalibDigiError other) {
  if (one.getCol() != other.getCol())
    return one.getCol() < other.getCol();

  return one.getRow() < other.getRow();
}
inline std::ostream& operator<<(std::ostream& output, const SiPixelCalibDigiError& err) {
  std::string errorstr = err.printError();
  return output << "pixel SiCalibDigi error in row " << err.getRow() << ", column " << err.getCol() << " of type "
                << err.getErrorType() << " (" << errorstr << ")";
}
#endif
