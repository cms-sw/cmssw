#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <cstring>

//
// Constructors
//
SiPixelGainCalibration::SiPixelGainCalibration() :
  minPed_(0.),
  maxPed_(255.),
  minGain_(0.),
  maxGain_(255.),
  numberOfRowsToAverageOver_(1),
  nBinsToUseForEncoding_(253),
  deadFlag_(255),
  noisyFlag_(254)
{
   if (deadFlag_ > 0xFF)
      throw cms::Exception("GainCalibration Payload configuration error")
         << "[SiPixelGainCalibration::SiPixelGainCalibration] Dead flag was set to " << deadFlag_ << ", and it must be set less than or equal to 255";
}
//
SiPixelGainCalibration::SiPixelGainCalibration(float minPed, float maxPed, float minGain, float maxGain) :
  minPed_(minPed),
  maxPed_(maxPed),
  minGain_(minGain),
  maxGain_(maxGain),
  numberOfRowsToAverageOver_(1),
  nBinsToUseForEncoding_(253),
  deadFlag_(255),
  noisyFlag_(254)
{
   if (deadFlag_ > 0xFF)
      throw cms::Exception("GainCalibration Payload configuration error")
         << "[SiPixelGainCalibration::SiPixelGainCalibration] Dead flag was set to " << deadFlag_ << ", and it must be set less than or equal to 255";
}

bool SiPixelGainCalibration::put(const uint32_t& DetId, Range input, const int& nCols, const int& ROCRows) {
  // put in SiPixelGainCalibration of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibration::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId)
    return false;
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ncols=nCols;
  detregistry.rocrows=ROCRows;
  detregistry.ibegin=v_pedestals.size();
  detregistry.iend=v_pedestals.size()+sd;
  indexes.insert(p,detregistry);

  v_pedestals.insert(v_pedestals.end(),input.first,input.second);
  return true;
}

const int SiPixelGainCalibration::getNCols(const uint32_t& DetId, int *ROCRows) const {
  // get number of columns of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibration::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return 0;
  else
    { if (ROCRows!=0) { *ROCRows=p->rocrows;}
    return p->ncols; 
    }
}

const SiPixelGainCalibration::Range SiPixelGainCalibration::getRange(const uint32_t& DetId) const {
  // get SiPixelGainCalibration Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibration::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiPixelGainCalibration::Range(v_pedestals.end(),v_pedestals.end()); 
  else 
    return SiPixelGainCalibration::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend);
}

const std::pair<const SiPixelGainCalibration::Range, const int>
SiPixelGainCalibration::getRangeAndNCols(const uint32_t& DetId, int* ROCRows) const {
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibration::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return std::make_pair(SiPixelGainCalibration::Range(v_pedestals.end(),v_pedestals.end()), 0); 
  else 
    { if (ROCRows!=0) { *ROCRows=p->rocrows;}
    return std::make_pair(SiPixelGainCalibration::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend), p->ncols);
    }
}
  

void SiPixelGainCalibration::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiPixelGainCalibration::RegistryIterator begin = indexes.begin();
  SiPixelGainCalibration::RegistryIterator end   = indexes.end();
  for (SiPixelGainCalibration::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiPixelGainCalibration::setData(float ped, float gain, std::vector<char>& vped, bool isDeadPixel, bool isNoisyPixel){
  
  float theEncodedGain=0;
  float theEncodedPed=0;
  if(!isDeadPixel && !isNoisyPixel){
    theEncodedPed = encodeGain(gain);
    theEncodedPed = encodePed (ped);
  }

  unsigned int ped_   = (static_cast<unsigned int>(theEncodedPed))  & 0xFF; 
  unsigned int gain_  = (static_cast<unsigned int>(theEncodedGain)) & 0xFF;

  if (isDeadPixel)
  {
     ped_  = deadFlag_ & 0xFF;
     gain_ = deadFlag_ & 0xFF;
  }
  if (isNoisyPixel)
  {
     ped_  = noisyFlag_ & 0xFF;
     gain_ = noisyFlag_ & 0xFF;
  }
  unsigned int data = (ped_ << 8) | gain_ ;
  vped.resize(vped.size()+2);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&data),2);
}

float SiPixelGainCalibration::getPed(const int& col, const int& row, const Range& range, const int& nCols, bool& isDead, bool& isNoisy, const int& ROCRows) const {

  int nRows = (range.second-range.first)/2 / nCols;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+(col*nRows + row)*2);
  if (col >= nCols || row >= nRows){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibration::getPed] Pixel out of range: col " << col << " row " << row;
  }  
  if ((s.ped & 0xFF) == deadFlag_)
     isDead = true;  
  if ((s.ped & 0xFF) == noisyFlag_)
     isNoisy = true;
  return decodePed(s.ped & 0xFF);  
}

float SiPixelGainCalibration::getGain(const int& col, const int& row, const Range& range, const int& nCols, bool& isDead, bool& isNoisy, const int& ROCRows) const {

  int nRows = (range.second-range.first)/2 / nCols;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+(col*nRows + row)*2);
  if (col >= nCols || row >= nRows){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibration::getPed] Pixel out of range: col " << col << " row " << row;
  }  
  if ((s.gain & 0xFF) == deadFlag_)
     isDead = true;  
  if ((s.gain & 0xFF) == noisyFlag_)
     isNoisy = true;
  return decodeGain(s.gain & 0xFF);
}

float SiPixelGainCalibration::encodeGain( const float& gain ) {
  
  if(gain < minGain_ || gain > maxGain_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibration::encodeGain] Trying to encode gain (" << gain << ") out of range [" << minGain_ << "," << maxGain_ << "]\n";
  } else {
    double precision   = (maxGain_-minGain_)/static_cast<float>(nBinsToUseForEncoding_);
    float  encodedGain = (float)((gain-minGain_)/precision);
    return encodedGain;
  }

}

float SiPixelGainCalibration::encodePed( const float& ped ) {

  if(ped < minPed_ || ped > maxPed_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibration::encodePed] Trying to encode pedestal (" << ped << ") out of range [" << minPed_ << "," << maxPed_ << "]\n";
  } else {
    double precision   = (maxPed_-minPed_)/static_cast<float>(nBinsToUseForEncoding_);
    float  encodedPed = (float)((ped-minPed_)/precision);
    return encodedPed;
  }

}

float SiPixelGainCalibration::decodePed( unsigned int ped ) const {

  double precision = (maxPed_-minPed_)/static_cast<float>(nBinsToUseForEncoding_);
  float decodedPed = (float)(ped*precision + minPed_);
  return decodedPed;

}

float SiPixelGainCalibration::decodeGain( unsigned int gain ) const {

  double precision = (maxGain_-minGain_)/static_cast<float>(nBinsToUseForEncoding_);
  float decodedGain = (float)(gain*precision + minGain_);
  return decodedGain;

}

