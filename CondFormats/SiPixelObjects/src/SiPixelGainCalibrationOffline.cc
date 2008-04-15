#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// Constructors
//
SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline() :
  minPed_(0.),
  maxPed_(254.),
  minGain_(0.),
  maxGain_(254.),
  nBins_(254.),
  deadVal_(255)
{
}
//
SiPixelGainCalibrationOffline::SiPixelGainCalibrationOffline(float minPed, float maxPed, float minGain, float maxGain) :
  minPed_(minPed),
  maxPed_(maxPed),
  minGain_(minGain),
  maxGain_(maxGain),
  nBins_(254.),
  deadVal_(255)
{
}

bool SiPixelGainCalibrationOffline::put(const uint32_t& DetId, Range input, const int& nCols) {
  // put in SiPixelGainCalibrationOffline of DetId

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationOffline::StrictWeakOrdering());
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

const int SiPixelGainCalibrationOffline::getNCols(const uint32_t& DetId) const {
  // get number of columns of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationOffline::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return 0;
  else
    return p->ncols; 
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
SiPixelGainCalibrationOffline::getRangeAndNCols(const uint32_t& DetId) const {
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiPixelGainCalibrationOffline::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return std::make_pair(SiPixelGainCalibrationOffline::Range(v_pedestals.end(),v_pedestals.end()), 0); 
  else 
    return std::make_pair(SiPixelGainCalibrationOffline::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend), p->ncols);
}
  

void SiPixelGainCalibrationOffline::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiPixelGainCalibrationOffline::RegistryIterator begin = indexes.begin();
  SiPixelGainCalibrationOffline::RegistryIterator end   = indexes.end();
  for (SiPixelGainCalibrationOffline::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}

void SiPixelGainCalibrationOffline::setDeadPixel(std::vector<char>& vped){
  float theEncodedPedestal  = deadVal_;

  unsigned int ped_  = (static_cast<unsigned int>(theEncodedPedestal)) & 0xFF;

  vped.resize(vped.size()+1);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-1]),(void*)(&ped_),1);
}
void SiPixelGainCalibrationOffline::setDeadCol(bool lowCol, bool highCol,const int &nRows,std::vector<char>& vped){
  
  float theEncodedGainLow  = deadVal_;
  float theEncodedGainHigh  = deadVal_;

  unsigned int gainLow_  = (static_cast<unsigned int>(theEncodedGainLow)) & 0xFF;
  unsigned int gainHigh_  = (static_cast<unsigned int>(theEncodedGainHigh)) & 0xFF;

  vped.resize(vped.size()+2);  //add a value on the end of this column for the high and low averages.

  //check to make sure the column is being placed in the right place in the blob
  if (vped.size() % (nRows + 2) != 0) 
  {
    throw cms::Exception("FillError")
      << "[SiPixelGainCalibrationOffline::setDataGain] Column gain average must be filled after the pedestal for each row has been added";
  }  
  // insert the two objects into the vector of chars 
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&gainLow_),1);
  ::memcpy((void*)(&vped[vped.size()-1]),(void*)(&gainHigh_),1);
}

void SiPixelGainCalibrationOffline::setDataGain(float gainLow, float gainHigh, const int& nRows, std::vector<char>& vped){
  
  float theEncodedGainLow  = encodeGain(gainLow);
  float theEncodedGainHigh  = encodeGain(gainHigh);

  unsigned int gainLow_  = (static_cast<unsigned int>(theEncodedGainLow)) & 0xFF;
  unsigned int gainHigh_  = (static_cast<unsigned int>(theEncodedGainHigh)) & 0xFF;

  vped.resize(vped.size()+2);  //add a value on the end of this column for the high and low averages.

  //check to make sure the column is being placed in the right place in the blob
  if (vped.size() % (nRows + 2) != 0) 
  {
    throw cms::Exception("FillError")
      << "[SiPixelGainCalibrationOffline::setDataGain] Column gain average must be filled after the pedestal for each row has been added";
  }  
  // insert the two objects into the vector of chars 
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&gainLow_),1);
  ::memcpy((void*)(&vped[vped.size()-1]),(void*)(&gainHigh_),1);
}

void SiPixelGainCalibrationOffline::setDataPedestal(float pedestal,  std::vector<char>& vped){

  float theEncodedPedestal  = encodePed(pedestal);

  unsigned int ped_  = (static_cast<unsigned int>(theEncodedPedestal)) & 0xFF;

  vped.resize(vped.size()+1);
  // insert in vector of char
  ::memcpy((void*)(&vped[vped.size()-1]),(void*)(&ped_),1);
}

float SiPixelGainCalibrationOffline::getPed(const int& col, const int& row, const Range& range, const int& nCols, bool & isDead) const {

  int lengthOfColumnData = (range.second-range.first)/nCols;

  isDead=false;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first + col*(lengthOfColumnData)+row);

  // ensure we aren't getting a nonsensical column, or retrieving one of the gain entries at the end of column stream
  if (col >= nCols || row >= lengthOfColumnData-2){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationOffline::getPed] Pixel out of range: col " << col << " row " << row;
  }  
  if(s.datum==deadVal_)
    isDead=true;
  return decodePed(s.datum & 0xFF);  
}

bool SiPixelGainCalibrationOffline::isDead(const int& col, const int& row, const Range& range, const int& nCols) const{
  int offset = 0; 
  if (row >= 80)
   offset = 1;
  int lengthOfColumnData = (range.second-range.first)/nCols;
  const DecodingStructure & sg = (const DecodingStructure & ) *(range.first+(col+1)*(lengthOfColumnData)-2 + offset);
  const DecodingStructure & sp = (const DecodingStructure & ) *(range.first + col*(lengthOfColumnData)+row);

  if (col >= nCols || row >= lengthOfColumnData-2){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationOffline::isDead] Pixel out of range: col " << col << " row " << row;
  } 
  if(sp.datum!=deadVal_ && sg.datum!=deadVal_)
    return false;
  else
    return true;

}
float SiPixelGainCalibrationOffline::getGain(const int& col, const int& row, const Range& range, const int& nCols, bool & isDead) const {

  //determine if we should get the low or high gain column average
  int offset = 0;
  isDead = false;
  if (row >= 80)
     offset = 1;
  int lengthOfColumnData = (range.second-range.first)/nCols;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+(col+1)*(lengthOfColumnData)-2 + offset);

  if (col >= nCols){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationOffline::getPed] Pixel out of range: col " << col;
  }  
  if(s.datum==deadVal_)
    isDead=true;
  return decodeGain(s.datum & 0xFF);
}

float SiPixelGainCalibrationOffline::encodeGain( const float& gain ) {
  
  if(gain < minGain_ || gain > maxGain_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationOffline::encodeGain] Trying to encode gain (" << gain << ") out of range [" << minGain_ << "," << maxGain_ << "]\n";
  } else {
    double precision   = (maxGain_-minGain_)/nBins_;
    float  encodedGain = (float)((gain-minGain_)/precision);
    return encodedGain;
  }

}

float SiPixelGainCalibrationOffline::encodePed( const float& ped ) {

  if(ped < minPed_ || ped > maxPed_ ) {
    throw cms::Exception("InsertFailure")
      << "[SiPixelGainCalibrationOffline::encodePed] Trying to encode pedestal (" << ped << ") out of range [" << minPed_ << "," << maxPed_ << "]\n";
  } else {
    double precision   = (maxPed_-minPed_)/nBins_;
    float  encodedPed = (float)((ped-minPed_)/precision);
    return encodedPed;
  }

}

float SiPixelGainCalibrationOffline::decodePed( unsigned int ped ) const {

  double precision = (maxPed_-minPed_)/nBins_;
  float decodedPed = (float)(ped*precision + minPed_);
  return decodedPed;

}

float SiPixelGainCalibrationOffline::decodeGain( unsigned int gain ) const {

  double precision = (maxGain_-minGain_)/nBins_;
  float decodedGain = (float)(gain*precision + minGain_);
  return decodedGain;

}


