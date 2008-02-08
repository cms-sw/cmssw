#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// Constructors
//
SiPixelGainCalibrationForHLT::SiPixelGainCalibrationForHLT() :
  minPed_(0.),
  maxPed_(255.),
  minGain_(0.),
  maxGain_(255.)
{
}
//
SiPixelGainCalibrationForHLT::SiPixelGainCalibrationForHLT(float minPed, float maxPed, float minGain, float maxGain) :
  minPed_(minPed),
  maxPed_(maxPed),
  minGain_(minGain),
  maxGain_(maxGain)
{
}

bool SiPixelGainCalibrationForHLT::put(const uint32_t& DetId, Range input) {
  // put in SiPixelGainCalibrationForHLT of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationForHLT::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId)
    return false;
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_pedestals.size();
  detregistry.iend=v_pedestals.size()+sd;
  indexes.insert(p,detregistry);

  v_pedestals.insert(v_pedestals.end(),input.first,input.second);
  return true;
}

const int SiPixelGainCalibrationForHLT::getNCols(const uint32_t& DetId) const {
  // get number of columns of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationForHLT::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return 0;
  else
  {
    int nCols = static_cast<int>(p->iend - p->ibegin);
    return nCols;
  }
}

const SiPixelGainCalibrationForHLT::Range SiPixelGainCalibrationForHLT::getRange(const uint32_t& DetId) const {
  // get SiPixelGainCalibrationForHLT Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationForHLT::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiPixelGainCalibrationForHLT::Range(v_pedestals.end(),v_pedestals.end()); 
  else 
    return SiPixelGainCalibrationForHLT::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend);
}

const std::pair<const SiPixelGainCalibrationForHLT::Range, const int>
SiPixelGainCalibrationForHLT::getRangeAndNCols(const uint32_t& DetId) const {
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationForHLT::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return std::make_pair(SiPixelGainCalibrationForHLT::Range(v_pedestals.end(),v_pedestals.end()), 0); 
  else 
    return std::make_pair(SiPixelGainCalibrationForHLT::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend), p->iend - p->ibegin);
}

void SiPixelGainCalibrationForHLT::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiPixelGainCalibrationForHLT::RegistryIterator begin = indexes.begin();
  SiPixelGainCalibrationForHLT::RegistryIterator end   = indexes.end();
  for (SiPixelGainCalibrationForHLT::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiPixelGainCalibrationForHLT::setData(float ped, float gain, std::vector<char>& vped){
  
  float theEncodedGain  = encodeGain(gain);
  float theEncodedPed   = encodePed (ped);

  unsigned int ped_   = (static_cast<unsigned int>(theEncodedPed))  & 0xFF; 
  unsigned int gain_  = (static_cast<unsigned int>(theEncodedGain)) & 0xFF;

  unsigned int data = (ped_ << 8) | gain_ ;
  vped.resize(vped.size()+2);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&data),2);
}

// DummyNCols only exists to preserve template compatability with other payloads
float SiPixelGainCalibrationForHLT::getPed(const int& col, const Range& range, const int& DummyNCols) const {

  int nCols = (range.second-range.first)/2;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+col*2);
  if (col >= nCols){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationForHLT::getPed] Pixel out of range: col " << col;
  }  
  return decodePed(s.ped & 0xFF);  
}

// DummyNCols only exists to preserve template compatability with other payloads
float SiPixelGainCalibrationForHLT::getGain(const int& col, const Range& range, const int& DummyNCols) const {

  int nCols = (range.second-range.first)/2;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+col*2);
  if (col >= nCols){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationForHLT::getGain] Pixel out of range: col " << col;
  }  
  return decodeGain(s.gain & 0xFF);  

}

float SiPixelGainCalibrationForHLT::encodeGain( const float& gain ) {
  
  if(gain < minGain_ || gain > maxGain_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationForHLT::encodeGain] Trying to encode gain (" << gain << ") out of range [" << minGain_ << "," << maxGain_ << "]\n";
  } else {
    double precision   = (maxGain_-minGain_)/255.;
    float  encodedGain = (float)((gain-minGain_)/precision);
    return encodedGain;
  }

}

float SiPixelGainCalibrationForHLT::encodePed( const float& ped ) {

  if(ped < minPed_ || ped > maxPed_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationForHLT::encodePed] Trying to encode pedestal (" << ped << ") out of range [" << minPed_ << "," << maxPed_ << "]\n";
  } else {
    double precision   = (maxPed_-minPed_)/255.;
    float  encodedPed = (float)((ped-minPed_)/precision);
    return encodedPed;
  }

}

float SiPixelGainCalibrationForHLT::decodePed( unsigned int ped ) const {

  double precision = (maxPed_-minPed_)/255.;
  float decodedPed = (float)(ped*precision + minPed_);
  return decodedPed;

}

float SiPixelGainCalibrationForHLT::decodeGain( unsigned int gain ) const {

  double precision = (maxGain_-minGain_)/255.;
  float decodedGain = (float)(gain*precision + minGain_);
  return decodedGain;

}


// functions for template compatibility with other payloads. should never run.
float SiPixelGainCalibrationForHLT::getGain(const int& col, const int& row, const Range& range, const int& nCols) const {
   throw cms::Exception("ConfigurationError")
      << "[SiPixelGainCalibration::getGain(col, row, range, ncols)] Data is stored at column granularity in this payload.  Please use getGain(col, range)";
   return -1.;
}

float SiPixelGainCalibrationForHLT::getPed(const int& col, const int& row, const Range& range, const int& nCols) const {
   throw cms::Exception("ConfigurationError")
      << "[SiPixelGainCalibration::getPed(col, row, range, ncols)] Data is stored at column granularity in this payload.  Please use getPed(col, range)";
   return -1.;
}
