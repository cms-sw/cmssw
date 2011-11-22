#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cstring>
#include <algorithm>

//
// Constructors
//
SiPixelGainCalibrationForHLT::SiPixelGainCalibrationForHLT() :
  minPed_(0.),
  maxPed_(255.),
  minGain_(0.),
  maxGain_(255.),
  numberOfRowsToAverageOver_(80),
  nBinsToUseForEncoding_(253),
  deadFlag_(255),
  noisyFlag_(254)
{
   if (deadFlag_ > 0xFF)
      throw cms::Exception("GainCalibration Payload configuration error")
         << "[SiPixelGainCalibrationHLT::SiPixelGainCalibrationHLT] Dead flag was set to " << deadFlag_ << ", and it must be set less than or equal to 255";
}
//
SiPixelGainCalibrationForHLT::SiPixelGainCalibrationForHLT(float minPed, float maxPed, float minGain, float maxGain) :
  minPed_(minPed),
  maxPed_(maxPed),
  minGain_(minGain),
  maxGain_(maxGain),
  numberOfRowsToAverageOver_(80),
  nBinsToUseForEncoding_(253),
  deadFlag_(255),
  noisyFlag_(254)
{
   if (deadFlag_ > 0xFF)
      throw cms::Exception("GainCalibration Payload configuration error")
         << "[SiPixelGainCalibrationHLT::SiPixelGainCalibrationHLT] Dead flag was set to " << deadFlag_ << ", and it must be set less than or equal to 255";
}

bool SiPixelGainCalibrationForHLT::put(const uint32_t& DetId, Range input, const int& nCols, const int& ROCRows) {
  // put in SiPixelGainCalibrationForHLT of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationForHLT::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId)
    return false;
  
  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_pedestals.size();
  detregistry.iend=v_pedestals.size()+sd;
  detregistry.ncols=nCols;
  detregistry.rocrows=ROCRows;
  indexes.insert(p,detregistry);

  v_pedestals.insert(v_pedestals.end(),input.first,input.second);
  return true;
}

const int SiPixelGainCalibrationForHLT::getNCols(const uint32_t& DetId, int *ROCRows) const {
  // get number of columns of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationForHLT::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return 0;
  else
  {  if (ROCRows!=0) { *ROCRows=p->rocrows;}
    return p->ncols;
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
SiPixelGainCalibrationForHLT::getRangeAndNCols(const uint32_t& DetId, int* ROCRows) const {
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationForHLT::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return std::make_pair(SiPixelGainCalibrationForHLT::Range(v_pedestals.end(),v_pedestals.end()), 0); 
  else 
    { if (ROCRows!=0) { *ROCRows=p->rocrows;}
    return std::make_pair(SiPixelGainCalibrationForHLT::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend), p->ncols);
    }
}

void SiPixelGainCalibrationForHLT::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiPixelGainCalibrationForHLT::RegistryIterator begin = indexes.begin();
  SiPixelGainCalibrationForHLT::RegistryIterator end   = indexes.end();
  for (SiPixelGainCalibrationForHLT::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiPixelGainCalibrationForHLT::setData(float ped, float gain, std::vector<char>& vped, bool thisColumnIsDead, bool thisColumnIsNoisy){
  
  float theEncodedGain=0;
  float theEncodedPed=0;
  if(!thisColumnIsDead && !thisColumnIsNoisy){
    theEncodedGain = encodeGain(gain);
    theEncodedPed = encodePed (ped);
  }

  unsigned int ped_   = (static_cast<unsigned int>(theEncodedPed))  & 0xFF; 
  unsigned int gain_  = (static_cast<unsigned int>(theEncodedGain)) & 0xFF;

  if (thisColumnIsDead)
  {
     ped_  = deadFlag_ & 0xFF;
     gain_ = deadFlag_ & 0xFF;
  }
  else if (thisColumnIsNoisy)
  {
     ped_  = noisyFlag_ & 0xFF;
     gain_ = noisyFlag_ & 0xFF;
  }

  unsigned int data = (ped_ << 8) | gain_ ;
  vped.resize(vped.size()+2);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&data),2);
}

float SiPixelGainCalibrationForHLT::getPed(const int& col, const int& row, const Range& range, const int& nCols, bool& isDeadColumn, bool& isNoisyColumn, const int& ROCRows) const {
   // TODO MERGE THIS FUNCTION WITH GET GAIN, then provide wrappers

  // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
  unsigned int lengthOfColumnData  = (range.second-range.first)/nCols;
  unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block 
  unsigned int numberOfDataBlocksToSkip = row / ROCRows;

  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+col*lengthOfColumnData + lengthOfAveragedDataInEachColumn*numberOfDataBlocksToSkip);

  if ((s.ped & 0xFF) == deadFlag_)
     isDeadColumn = true;
  else if ((s.ped & 0xFF) == noisyFlag_)
     isNoisyColumn = true;

  int maxRow = (lengthOfColumnData/lengthOfAveragedDataInEachColumn)*ROCRows - 1;
  if (col >= nCols || row > maxRow){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationForHLT::getPed] Pixel out of range: col " << col << " row: " << row;
  }  
  return decodePed(s.ped & 0xFF);  
}

float SiPixelGainCalibrationForHLT::getGain(const int& col, const int& row, const Range& range, const int& nCols, bool& isDeadColumn, bool& isNoisyColumn, const int& ROCRows) const {

  // determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
  unsigned int lengthOfColumnData  = (range.second-range.first)/nCols;
  unsigned int lengthOfAveragedDataInEachColumn = 2;  // we always only have two values per column averaged block 
  unsigned int numberOfDataBlocksToSkip = row / ROCRows;

  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+col*lengthOfColumnData + lengthOfAveragedDataInEachColumn*numberOfDataBlocksToSkip);

  if ((s.gain & 0xFF) == deadFlag_)
     isDeadColumn = true;
  else if ((s.gain & 0xFF) == noisyFlag_)
     isNoisyColumn = true;
     
  int maxRow = (lengthOfColumnData/lengthOfAveragedDataInEachColumn)*ROCRows - 1;
  if (col >= nCols || row > maxRow){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationForHLT::getGain] Pixel out of range: col " << col << " row: " << row;
  }  
  return decodeGain(s.gain & 0xFF);  

}

float SiPixelGainCalibrationForHLT::encodeGain( const float& gain ) {
  
  if(gain < minGain_ || gain > maxGain_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationForHLT::encodeGain] Trying to encode gain (" << gain << ") out of range [" << minGain_ << "," << maxGain_ << "]\n";
  } else {
    double precision   = (maxGain_-minGain_)/static_cast<float>(nBinsToUseForEncoding_);
    float  encodedGain = (float)((gain-minGain_)/precision);
    return encodedGain;
  }

}

float SiPixelGainCalibrationForHLT::encodePed( const float& ped ) {

  if(ped < minPed_ || ped > maxPed_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationForHLT::encodePed] Trying to encode pedestal (" << ped << ") out of range [" << minPed_ << "," << maxPed_ << "]\n";
  } else {
    double precision   = (maxPed_-minPed_)/static_cast<float>(nBinsToUseForEncoding_);
    float  encodedPed = (float)((ped-minPed_)/precision);
    return encodedPed;
  }

}

float SiPixelGainCalibrationForHLT::decodePed( unsigned int ped ) const {

  double precision = (maxPed_-minPed_)/static_cast<float>(nBinsToUseForEncoding_);
  float decodedPed = (float)(ped*precision + minPed_);
  return decodedPed;

}

float SiPixelGainCalibrationForHLT::decodeGain( unsigned int gain ) const {

  double precision = (maxGain_-minGain_)/static_cast<float>(nBinsToUseForEncoding_);
  float decodedGain = (float)(gain*precision + minGain_);
  return decodedGain;

}


