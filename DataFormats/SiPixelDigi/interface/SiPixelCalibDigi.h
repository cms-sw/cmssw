#ifndef DATAFORMATS_SIPIXELCALIBDIGI_SIPIXELCALIBDIGI_H
#define DATAFORMATS_SIPIXELCALIBDIGI_SIPIXELCALIBDIGI_H

#include <utility>
#include <vector>
#include <iostream>
#include <cstdint>

class SiPixelCalibDigi {
public:
  SiPixelCalibDigi(int packed_value, uint16_t row, uint16_t col) : m_data(packed_value), m_row_and_column(row, col) {
    reset();
  }
  SiPixelCalibDigi(int packed_value) : m_data(packed_value), m_row_and_column(0, 0) { reset(); }
  SiPixelCalibDigi() : m_data(0), m_row_and_column(0, 0) { ; }
  virtual ~SiPixelCalibDigi() { ; }

  void init(uint16_t row, uint16_t col, uint16_t npoints);
  void reset();
  void fill(uint32_t ipoint, uint32_t adcval);
  void setrowcol(uint16_t row, uint16_t col);

  // analysis methods:
  uint32_t getnpoints() const { return m_data.size(); }  // returns the number of calibration points
  uint32_t getsum(uint32_t ipoint) const {
    return m_data[ipoint].adccountsum;
  }  // returns the sum of adc counts for the collected events at point ipoint
  uint32_t getsumsquares(uint32_t ipoint) const {
    return m_data[ipoint].adccountsumsq;
  }  // returns the sum of the squares of adc counts for the collected events at point ipoint
  uint32_t getnentries(uint32_t ipoint) const {
    return m_data[ipoint].nentries;
  }                                             // returns the number of entries made for calibration point ipoint
  std::vector<uint32_t> getsum() const;         // returns the sum of adc counts for the collected events for all points
  std::vector<uint32_t> getsumsquares() const;  // returns the sum of the squares of adc counts
  std::vector<uint8_t> getnentries() const;     // returns the number of entries made
  uint16_t row() const { return m_row_and_column.first; }
  uint16_t col() const { return m_row_and_column.second; }
  std::pair<uint16_t, uint16_t> row_and_col() const { return m_row_and_column; }

  struct datacontainer {
    datacontainer() : nentries(0), adccountsum(0), adccountsumsq(0) {}
    uint8_t nentries;        // the number of entries per VCAL point
    uint32_t adccountsum;    // the sum of the ADC counts
    uint32_t adccountsumsq;  // the sum of the square value of the ADC counts
  };

private:
  // this is were the data is stored
  std::vector<datacontainer> m_data;
  std::pair<uint16_t, uint16_t> m_row_and_column;  //the row and column number of this particular pixel
};

inline bool operator<(const SiPixelCalibDigi& one, const SiPixelCalibDigi& other) {
  return one.getnpoints() < other.getnpoints();
}

inline std::ostream& operator<<(std::ostream& output, const SiPixelCalibDigi& digi) {
  /*   std::ostream result; */
  /*   result << "this SiPixelCalibDigi object contains: " << m_nentries.size() << " and represents pixel (row,col) (" << m_row_and_column.first << ","<< m_row_and_colum.second <<") \n"; */
  /*   for(int iter=0; iter<m_nentries.size(); ++iter) */
  /*     result << "calib point " << iter << ", nentries " << m_nentries[iter] << ", sum " << m_adccountsum[iter] << ", sum of squares " << m_adccountsumsq[iter] <<".\n"; */
  /*   return output << result; */
  return output << "this pixel (row " << digi.row() << ", col " << digi.col() << " has " << digi.getnpoints()
                << " calibration points.";
}

#endif
