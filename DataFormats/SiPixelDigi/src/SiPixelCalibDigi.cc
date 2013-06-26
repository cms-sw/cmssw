
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"

#include <algorithm>

// the actual data container is a simply a collection of SiPixelCalibDigi objects

// init method, sets array sizes to npoints, fixes number of points 
void SiPixelCalibDigi::init( uint16_t row, uint16_t col, uint16_t npoints) {
  setrowcol(row,col);
  reset();
  // expand if size of arrays has changed.... not implemented yet...
}
// reset method, sets all array values to zero, does not change size
void SiPixelCalibDigi::reset(){
  for(uint32_t iter =0; iter<m_data.size();++iter){
    m_data[iter].nentries=0;
    m_data[iter].adccountsum=0;
    m_data[iter].adccountsumsq=0;
  }
}
// fill method, adds information to the arrays
void SiPixelCalibDigi::fill(uint32_t ipoint, uint32_t adcval) {
  if(ipoint>=m_data.size())
    return;
  m_data[ipoint].nentries++;
  m_data[ipoint].adccountsum+=adcval;
  m_data[ipoint].adccountsumsq+=adcval*adcval;
  return;
}
// set method to rol row and column number
void SiPixelCalibDigi::setrowcol(uint16_t row, uint16_t col){
  m_row_and_column.first=row;
  m_row_and_column.second=col;
}
// possibility to retrieve vector of entry objects
std::vector<uint8_t> SiPixelCalibDigi::getnentries() const{
  std::vector<uint8_t> result(0);
  for(uint32_t ii=0; ii<m_data.size();++ii)
    result.push_back(m_data[ii].nentries);
  return result;
}
// possibility to retrieve a vector of sums
std::vector<uint32_t> SiPixelCalibDigi::getsum() const{
  std::vector<uint32_t> result(0);
  for(uint32_t ii=0; ii<m_data.size();++ii)
    result.push_back(m_data[ii].adccountsum);
  return result;
}
// possibility to retrieve a vector of sumsquares
std::vector<uint32_t> SiPixelCalibDigi::getsumsquares() const{
  std::vector<uint32_t> result(0);
  for(uint32_t ii=0; ii<m_data.size();++ii)
    result.push_back(m_data[ii].adccountsumsq);
  return result;
}



