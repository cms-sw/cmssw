#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <cstring>

//	
// Constructors
//
SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline() :
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
         << "[SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline] Dead flag was set to " << deadFlag_ << ", and it must be set less than or equal to 255";
   if (noisyFlag_ > 0xFF)
      throw cms::Exception("GainCalibration Payload configuration error")
         << "[SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline] Noisy flag was set to " << noisyFlag_ << ", and it must be set less than or equal to 255";
}
//
SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline(float minPed, float maxPed, float minGain, float maxGain) :
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
         << "[SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline] Dead flag was set to " << deadFlag_ << ", and it must be set less than or equal to 255";
   if (noisyFlag_ > 0xFF)
      throw cms::Exception("GainCalibration Payload configuration error")
         << "[SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline] Noisy flag was set to " << noisyFlag_ << ", and it must be set less than or equal to 255";
}

bool SiPixelGainCalibrationOffline::put(const uint32_t& DetId, Range input, const int& nCols, const int& ROCRows) {
  // put in SiPixelGainCalibrationOffline of DetId
  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationOffline::StrictWeakOrdering());
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

const int SiPixelGainCalibrationOffline::getNCols(const uint32_t& DetId, int *ROCRows) const {
  // get number of columns of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationOffline::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return 0;
  else
    { if (ROCRows!=0) { *ROCRows=p->rocrows;}
    return p->ncols; 
    }
}

const SiPixelGainCalibrationOffline::Range SiPixelGainCalibrationOffline::getRange(const uint32_t& DetId) const {
  // get SiPixelGainCalibrationOffline Range of DetId
  
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationOffline::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiPixelGainCalibrationOffline::Range(v_pedestals.end(),v_pedestals.end()); 
  else 
    return SiPixelGainCalibrationOffline::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend);
}

const std::pair<const SiPixelGainCalibrationOffline::Range, const int>
SiPixelGainCalibrationOffline::getRangeAndNCols(const uint32_t& DetId, int* ROCRows) const {
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationOffline::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    { 
    return std::make_pair(SiPixelGainCalibrationOffline::Range(v_pedestals.end(),v_pedestals.end()), 0); 
    }
  else 
    { if (ROCRows!=0) { *ROCRows=p->rocrows;}
    return std::make_pair(SiPixelGainCalibrationOffline::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend), p->ncols);
    }
}
  

void SiPixelGainCalibrationOffline::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiPixelGainCalibrationOffline::RegistryIterator begin = indexes.begin();
  SiPixelGainCalibrationOffline::RegistryIterator end   = indexes.end();
  for (SiPixelGainCalibrationOffline::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiPixelGainCalibrationOffline::setDataGain(float gain, const int& nRows, std::vector<char>& vped, bool thisColumnIsDead, bool thisColumnIsNoisy){
  
  float theEncodedGain=0;
  if(!thisColumnIsDead && !thisColumnIsNoisy)
    theEncodedGain = encodeGain(gain);

  unsigned int gain_  = (static_cast<unsigned int>(theEncodedGain)) & 0xFF;

  // if this whole column is dead, set a char based dead flag in the blob.
  if (thisColumnIsDead)
     gain_ = deadFlag_ & 0xFF;
  if (thisColumnIsNoisy)
     gain_ = noisyFlag_ & 0xFF;

  vped.resize(vped.size()+1);
  //check to make sure the column is being placed in the right place in the blob
  /*  Remove this test for now.  Perhaps need to make a setter for numberOfRowsToAverageOver_?  idr 3/11/09
      if (nRows != (int)numberOfRowsToAverageOver_)
  {
    throw cms::Exception("GainCalibration Payload configuration error")
      << "[SiPixelGainCalibrationOffline::setDataGain] You are setting a gain averaged over nRows = " << nRows << " where this payload is set ONLY to average over " << numberOfRowsToAverageOver_ << " nRows";
  }

  */
  //  std::cout<<"setDataGain: "<<vped.size()<<" "<<nRows<<std::endl;

  if (vped.size() % (nRows + 1) != 0) 
  {
    throw cms::Exception("FillError")
      << "[SiPixelGainCalibrationOffline::setDataGain] Column gain average (OR SETTING AN ENTIRE COLUMN DEAD/NOISY) must be filled after the pedestal for each row has been added. An additional source of this error would be setting a pixel dead/noisy AND setting its pedestal";
  }  
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-1]),(void*)(&gain_),1);
}

void SiPixelGainCalibrationOffline::setDataPedestal(float pedestal,  std::vector<char>& vped, bool thisPixelIsDead, bool thisPixelIsNoisy){

  float theEncodedPedestal  = encodePed(pedestal);

  unsigned int ped_  = (static_cast<unsigned int>(theEncodedPedestal)) & 0xFF;

  if (thisPixelIsDead)
     ped_ = deadFlag_ & 0xFF;
  if (thisPixelIsNoisy)
     ped_ = noisyFlag_ & 0xFF;

  vped.resize(vped.size()+1);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-1]),(void*)(&ped_),1);
}

float SiPixelGainCalibrationOffline::getPed(const int& col, const int& row, const Range& range, const int& nCols, bool& isDead, bool& isNoisy, const int& ROCRows) const {

  unsigned int lengthOfColumnData = (range.second-range.first)/nCols;
  //determine what row averaged range we are in (i.e. ROC 1 or ROC 2)
  unsigned int lengthOfAveragedDataInEachColumn = ROCRows + 1;
  unsigned int numberOfAveragedDataBlocksToSkip = row / ROCRows;
  unsigned int offSetInCorrectDataBlock         = row % ROCRows;

  const DecodingStructure & s = (const DecodingStructure & ) *(range.first + col*(lengthOfColumnData) + (numberOfAveragedDataBlocksToSkip * lengthOfAveragedDataInEachColumn) + offSetInCorrectDataBlock);

  int maxRow = lengthOfColumnData - (lengthOfColumnData % ROCRows) - 1;
  if (col >= nCols || row > maxRow){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationOffline::getPed] Pixel out of range: col " << col << " row " << row;
  }  

  if ((s.datum & 0xFF) == deadFlag_)
     isDead = true;
  if ((s.datum & 0xFF) == noisyFlag_)
     isNoisy = true;

  return decodePed(s.datum & 0xFF);  
}

float SiPixelGainCalibrationOffline::getGain(const int& col, const int& row, const Range& range, const int& nCols, bool& isDeadColumn, bool& isNoisyColumn, const int& ROCRows) const {

  unsigned int lengthOfColumnData = (range.second-range.first)/nCols;
  //determine what row averaged range we are in (i.e. ROC 1 or ROC 2)
  unsigned int lengthOfAveragedDataInEachColumn = ROCRows + 1;
  unsigned int numberOfAveragedDataBlocksToSkip = row / ROCRows;

  // gain average is stored in the last location of current row averaged column data block
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+(col)*(lengthOfColumnData) + ( (numberOfAveragedDataBlocksToSkip+1) * lengthOfAveragedDataInEachColumn) - 1);

  if ((s.datum & 0xFF) == deadFlag_)
     isDeadColumn = true;
  if ((s.datum & 0xFF) == noisyFlag_)
     isNoisyColumn = true;

  int maxRow = lengthOfColumnData - (lengthOfColumnData % ROCRows) - 1;
  if (col >= nCols || row > maxRow){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationOffline::getPed] Pixel out of range: col " << col;
  }  
  return decodeGain(s.datum & 0xFF);
}

float SiPixelGainCalibrationOffline::encodeGain( const float& gain ) {
  
  if(gain < minGain_ || gain > maxGain_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationOffline::encodeGain] Trying to encode gain (" << gain << ") out of range [" << minGain_ << "," << maxGain_ << "]\n";
  } else {
    double precision   = (maxGain_-minGain_)/static_cast<float>(nBinsToUseForEncoding_);
    float  encodedGain = (float)((gain-minGain_)/precision);
    return encodedGain;
  }

}

float SiPixelGainCalibrationOffline::encodePed( const float& ped ) {

  if(ped < minPed_ || ped > maxPed_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationOffline::encodePed] Trying to encode pedestal (" << ped << ") out of range [" << minPed_ << "," << maxPed_ << "]\n";
  } else {
    double precision   = (maxPed_-minPed_)/static_cast<float>(nBinsToUseForEncoding_);
    float  encodedPed = (float)((ped-minPed_)/precision);
    return encodedPed;
  }

}

float SiPixelGainCalibrationOffline::decodePed( unsigned int ped ) const {

  double precision = (maxPed_-minPed_)/static_cast<float>(nBinsToUseForEncoding_);
  float decodedPed = (float)(ped*precision + minPed_);
  return decodedPed;

}

float SiPixelGainCalibrationOffline::decodeGain( unsigned int gain ) const {

  double precision = (maxGain_-minGain_)/static_cast<float>(nBinsToUseForEncoding_);
  float decodedGain = (float)(gain*precision + minGain_);
  return decodedGain;
}

