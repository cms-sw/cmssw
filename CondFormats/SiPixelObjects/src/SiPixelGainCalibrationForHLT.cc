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
    int nCols = static_cast<int>(p->iend - p->ibegin)/4;
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

void SiPixelGainCalibrationForHLT::setData(float pedLowRows, float gainLowRows, float pedHighRows, float gainHighRows, std::vector<char>& vped){
  
  // Blob encoding scheme:
  // Each column has four chars, arranged like: [ pedAvgLowRow - gainAvgLowRow - pedAvgHighRow - gainAvgHighRow ]
  // Note that for plaquettes with only one row of ROCS, the high rows must be filled so that
  // the blob is parsed correctly when it is read out.  FAILURE TO DO SO WILL RESULT IN CORRUPT OUTPUT 

  float theEncodedGainLowRows  = encodeGain(gainLowRows);
  float theEncodedPedLowRows   = encodePed (pedLowRows);

  float theEncodedGainHighRows  = encodeGain(gainHighRows);
  float theEncodedPedHighRows   = encodePed (pedHighRows);

  unsigned int pedLow_   = (static_cast<unsigned int>(theEncodedPedLowRows))  & 0xFF; 
  unsigned int gainLow_  = (static_cast<unsigned int>(theEncodedGainLowRows)) & 0xFF;

  unsigned int pedHigh_   = (static_cast<unsigned int>(theEncodedPedHighRows))  & 0xFF; 
  unsigned int gainHigh_  = (static_cast<unsigned int>(theEncodedGainHighRows)) & 0xFF;

  unsigned int dataLow = (pedLow_ << 8) | gainLow_ ;
  vped.resize(vped.size()+2);
  // insert in low columns data
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&dataLow),2);

  unsigned int dataHigh = (pedHigh_ << 8) | gainHigh_ ;
  vped.resize(vped.size()+2);
  // insert in high columns data
  ::memcpy((void*)(&vped[vped.size()-2]),(void*)(&dataHigh),2);
}

// DummyNCols only exists to preserve template compatability with other payloads
float SiPixelGainCalibrationForHLT::getPed(const int& col, const int& row, const Range& range, const int& DummyNCols) const {

  int nCols = (range.second-range.first)/4;
  int offset = 0;
  // check if this is a high row and adjust offset accordingly
  if (row >= 80)
     offset = 2;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+col*4 + offset);
  if (col >= nCols){
    throw cms::Exception("CorruptedData")
      << "[SiPixelGainCalibrationForHLT::getPed] Pixel out of range: col " << col;
  }  
  return decodePed(s.ped & 0xFF);  
}

// DummyNCols only exists to preserve template compatability with other payloads
//
float SiPixelGainCalibrationForHLT::getGain(const int& col, const int& row, const Range& range, const int& DummyNCols) const {

  int nCols = (range.second-range.first)/4;
  int offset = 0;
  // check if this is a high row and adjust offset accordingly
  if (row >= 80)
     offset = 2;
  const DecodingStructure & s = (const DecodingStructure & ) *(range.first+col*4 + offset);
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


