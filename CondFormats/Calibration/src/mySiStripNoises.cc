#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include <algorithm>
bool mySiStripNoises::put(const uint32_t DetId, InputVector& input) {
  // put in SiStripNoises of DetId
  std::vector<unsigned char>	Vo_CHAR;
  encode(input, Vo_CHAR);
  Registry::iterator p=std::lower_bound(indexes.begin(),indexes.end(),DetId,mySiStripNoises::StrictWeakOrdering());
  if (p!=indexes.end() 	&& 	p->detid==DetId)
    return false;
  size_t sd = Vo_CHAR.end() - Vo_CHAR.begin();
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_noises.size();
  detregistry.iend=v_noises.size()+sd;
  indexes.insert(p,detregistry);
  v_noises.insert(v_noises.end(),Vo_CHAR.begin(),Vo_CHAR.end());
  return true;
}

const mySiStripNoises::Range mySiStripNoises::getRange(const uint32_t& DetId) const {
  // get SiStripNoises Range of DetId
  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return mySiStripNoises::Range(v_noises.end(),v_noises.end()); 
  else 
    return mySiStripNoises::Range(v_noises.begin()+p->ibegin,v_noises.begin()+p->iend);
}

void mySiStripNoises::getDetIds(std::vector<uint32_t>& DetIds) const {
  // returns vector of DetIds in map
  mySiStripNoises::RegistryIterator begin = indexes.begin();
  mySiStripNoises::RegistryIterator end   = indexes.end();
  for (mySiStripNoises::RegistryIterator p=begin; p != end; ++p) {
    DetIds.push_back(p->detid);
  }
}

float mySiStripNoises::getNoise( const uint16_t& strip, const Range& range ) const{
  return   static_cast<float> (decode(strip,range)/10.0);
} 

void mySiStripNoises::setData(float noise_, std::vector<short>& v){
  v.push_back((static_cast<int16_t>  (noise_*10.0 + 0.5) & 0x01FF)) ;
}

void mySiStripNoises::encode(InputVector& Vi, std::vector<unsigned char>& Vo){
  static const uint16_t  BITS_PER_STRIP  = 9;
  const size_t           VoSize          = (size_t)((Vi.size() *       BITS_PER_STRIP)/8+.999);
  Vo.resize(VoSize);
  for(size_t i = 0; i<Vo.size(); ++i)
    Vo[i]   &=      0x00u;
  
  for(unsigned int stripIndex =0; stripIndex<Vi.size(); ++stripIndex){
    unsigned char*  data    =       &Vo[Vo.size()-1];
    uint32_t lowBit         =       stripIndex * BITS_PER_STRIP;
    uint8_t firstByteBit    =       (lowBit & 0x7);
    uint8_t firstByteNBits  =       8 - firstByteBit;
    uint8_t firstByteMask   =       0xffu << firstByteBit;
    uint8_t secondByteNbits =       (BITS_PER_STRIP - firstByteNBits);
    uint8_t secondByteMask  =       ~(0xffu << secondByteNbits);

    *(data-lowBit/8)        =       (*(data-lowBit/8)   & ~(firstByteMask))         | ((Vi[stripIndex] & 0xffu) <<firstByteBit);
    *(data-lowBit/8-1)      =       (*(data-lowBit/8-1) & ~(secondByteMask))        | ((Vi[stripIndex] >> firstByteNBits) & secondByteMask);
  }
}

uint16_t mySiStripNoises::decode (const uint16_t& strip, const Range& range) const{
  const unsigned char *data = &*(range.second -1);  // pointer to the last byte of data
  static const uint16_t BITS_PER_STRIP = 9;
  
  uint32_t lowBit        = strip * BITS_PER_STRIP;
  uint8_t firstByteBit   = (lowBit & 7);//module 8
  uint8_t firstByteNBits = 8 - firstByteBit;
  uint8_t firstByteMask  = 0xffu << firstByteBit;
  uint8_t secondByteMask = ~(0xffu << (BITS_PER_STRIP - firstByteNBits));
  uint16_t value         =   ((uint16_t(*(data-lowBit/8  )) & firstByteMask) >> firstByteBit) | ((uint16_t(*(data-lowBit/8-1)) & secondByteMask) << firstByteNBits);
  return value;
}


