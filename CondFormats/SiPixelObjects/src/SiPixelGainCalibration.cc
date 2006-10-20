#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "FWCore/Utilities/interface/Exception.h"

bool SiPixelGainCalibration::put(const uint32_t& DetId, Range input, const int& nCols) {
  // put in SiPixelGainCalibration of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibration::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId)
    return false;
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ncols=nCols;
  detregistry.ibegin=v_pedestals.size();
  detregistry.iend=v_pedestals.size()+sd;
  indexes.insert(p,detregistry);

  v_pedestals.insert(v_pedestals.end(),input.first,input.second);
  return true;
}

const int SiPixelGainCalibration::getNCols(const uint32_t& DetId) const {
  // get number of columns of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibration::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return 0;
  else
    return p->ncols; 
}

const SiPixelGainCalibration::Range SiPixelGainCalibration::getRange(const uint32_t& DetId) const {
  // get SiPixelGainCalibration Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibration::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiPixelGainCalibration::Range(v_pedestals.end(),v_pedestals.end()); 
  else 
    return SiPixelGainCalibration::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend);
}

void SiPixelGainCalibration::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiPixelGainCalibration::RegistryIterator begin = indexes.begin();
  SiPixelGainCalibration::RegistryIterator end   = indexes.end();
  for (SiPixelGainCalibration::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiPixelGainCalibration::setData(float ped, float gain, std::vector<char>& vped){
  
  // ACTION: Should ped and gain be rescaled to match the 8bit dinamic range?

  unsigned int ped_   = (static_cast<unsigned int>(ped))  & 0xFF; 
  unsigned int gain_  = (static_cast<unsigned int>(gain)) & 0xFF;

  //  unsigned int low_  = (static_cast<unsigned int>(lth*5.0+0.5)) & 0x3F; 
  // unsigned int hig_  = (static_cast<unsigned int>(hth*5.0+0.5)) & 0x3F; 
  //  unsigned int data = (ped_ << 12) | (hig_ << 6) | low_ ;

  unsigned int data = (ped_ << 8) | gain_ ;
  vped.resize(vped.size()+2);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&data),2);
}

float SiPixelGainCalibration::getPed(const int& col, const int& row, const Range& range, const int& nCols) const {

  int nRows = (range.second-range.first)/2 / nCols;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+(col*nRows + row)*2);
  if (col >= nCols || row >= nRows){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibration::getPed] Pixel out of range: col " << col << " row " << row;
  }  
  return (s.ped & 0xFF);  
}

float SiPixelGainCalibration::getGain(const int& col, const int& row, const Range& range, const int& nCols) const {

  int nRows = (range.second-range.first)/2 / nCols;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+(col*nRows + row)*2);
  if (col >= nCols || row >= nRows){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibration::getPed] Pixel out of range: col " << col << " row " << row;
  }  
  return (s.gain & 0xFF);
}
