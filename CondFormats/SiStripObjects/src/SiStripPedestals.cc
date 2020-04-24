#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>

bool SiStripPedestals::put(const uint32_t& DetId, InputVector& input) {
  // put in SiStripPedestals of DetId
  std::vector<unsigned char>      Vo_CHAR;
  encode(input, Vo_CHAR);

  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripPedestals::StrictWeakOrdering());
  if (p!=indexes.end() && p->detid==DetId)
    return false;

  //size_t sd= input.second-input.first;
  size_t sd= Vo_CHAR.end() - Vo_CHAR.begin();
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_pedestals.size();
  detregistry.iend=v_pedestals.size()+sd;
  indexes.insert(p,detregistry);

  //v_pedestals.insert(v_pedestals.end(),input.first,input.second);
  v_pedestals.insert(v_pedestals.end(),Vo_CHAR.begin(),Vo_CHAR.end());
  return true;
}

const SiStripPedestals::Range SiStripPedestals::getRange(const uint32_t& DetId) const {
  // get SiStripPedestals Range of DetId

  RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripPedestals::StrictWeakOrdering());
  if (p==indexes.end()|| p->detid!=DetId) 
    return SiStripPedestals::Range(v_pedestals.end(),v_pedestals.end()); 
  else 
    return SiStripPedestals::Range(v_pedestals.begin()+p->ibegin,v_pedestals.begin()+p->iend);
}

void SiStripPedestals::getDetIds(std::vector<uint32_t>& DetIds_) const {
  // returns vector of DetIds in map
  SiStripPedestals::RegistryIterator begin = indexes.begin();
  SiStripPedestals::RegistryIterator end   = indexes.end();
  for (SiStripPedestals::RegistryIterator p=begin; p != end; ++p) {
    DetIds_.push_back(p->detid);
  }
}


void SiStripPedestals::setData(float ped, SiStripPedestals::InputVector& vped){
  vped.push_back((static_cast<uint16_t>  (ped) & 0x3FF)) ;
}

float SiStripPedestals::getPed(const uint16_t& strip, const Range& range) const {
  if (10*strip>=(range.second-range.first)*8){
    throw cms::Exception("CorruptedData")
      << "[SiStripPedestals::getPed] looking for SiStripPedestals for a strip out of range: strip " << strip;
  }
  return   static_cast<float> (decode(strip,range));
}

void SiStripPedestals::encode(InputVector& Vi, std::vector<unsigned char>& Vo){
  static const uint16_t  BITS_PER_STRIP  = 10;
  const size_t           VoSize          = (size_t)((Vi.size() *       BITS_PER_STRIP)/8+.999);
  Vo.resize(VoSize);
  for(size_t i = 0; i<Vo.size(); ++i)
    Vo[i]   &=      0x00u;
  
  for(unsigned int stripIndex =0; stripIndex<Vi.size(); ++stripIndex){
    unsigned char*  data    =       &Vo[Vo.size()-1];
    uint32_t lowBit         =       stripIndex * BITS_PER_STRIP;
    uint8_t firstByteBit    =       (lowBit & 0x6);
    uint8_t firstByteNBits  =       8 - firstByteBit;
    uint8_t firstByteMask   =       0xffu << firstByteBit;
    uint8_t secondByteNbits =       (BITS_PER_STRIP - firstByteNBits);
    uint8_t secondByteMask  =       ~(0xffu << secondByteNbits);

    *(data-lowBit/8)        =       (*(data-lowBit/8)   & ~(firstByteMask))         | ((Vi[stripIndex] & 0xffu) <<firstByteBit);
    *(data-lowBit/8-1)      =       (*(data-lowBit/8-1) & ~(secondByteMask))        | ((Vi[stripIndex] >> firstByteNBits) & secondByteMask);

    /*
      if(stripIndex   < 25 ){
      std::cout       << "***************ENCODE*********************"<<std::endl
      << "\tdata-lowBit/8     :"<<print_as_binary((*(data-lowBit/8)   & ~(firstByteMask)))
      << "-"<<print_as_binary(((Vi[stripIndex] & 0xffu) <<firstByteBit))
      << "\tdata-lowBit/8-1   :"<<print_as_binary((*(data-lowBit/8-1)   & ~(secondByteMask)))
      << "-"<<print_as_binary((((Vi[stripIndex]>> firstByteNBits) & secondByteMask)))
      << std::endl;
      std::cout       << "strip "<<stripIndex<<"\tvi: " << Vi[stripIndex] <<"\t"
      << print_short_as_binary(Vi[stripIndex])
      << "\tvo1:"<< print_char_as_binary(*(data-lowBit/8))
      << "\tvo2:"<< print_char_as_binary(*(data-lowBit/8-1))
      << "\tlowBit:"<< lowBit
      << "\tfirstByteMask :"<<print_as_binary(firstByteMask)
      << "\tsecondByteMask:"<<print_as_binary(secondByteMask)
      << "\tfirstByteBit:"<<print_as_binary(firstByteBit)
      << std::endl;
      }
    */
  }
}

uint16_t SiStripPedestals::decode (const uint16_t& strip, const Range& range) const{
  const char *data = &*(range.second -1);  // pointer to the last byte of data
  static const uint16_t BITS_PER_STRIP = 10;

  uint32_t lowBit        = strip * BITS_PER_STRIP;
  uint8_t firstByteBit   = (lowBit & 6);//module 
  uint8_t firstByteNBits = 8 - firstByteBit;
  uint8_t firstByteMask  = 0xffu << firstByteBit;
  uint8_t secondByteMask = ~(0xffu << (BITS_PER_STRIP - firstByteNBits));
  uint16_t value         =   ((uint16_t(*(data-lowBit/8  )) & firstByteMask) >> firstByteBit) | ((uint16_t(*(data-lowBit/8-1)) & secondByteMask) << firstByteNBits);
  
  /*
    if(strip  < 25){
    std::cout       << "***************DECODE*********************"<<"\n"
    << "strip "<<strip << " " 
    << value 
    <<"\t   :"<<print_as_binary(value) 
    <<"\t  :"<<print_as_binary(    ((uint16_t(*(data-lowBit/8  )) & firstByteMask) >>   firstByteBit)       )
    << "-"<<print_as_binary(  ((uint16_t(*(data-lowBit/8-1)) & secondByteMask) <<firstByteNBits)    )
    << "\t *(data-lowBit/8) " << print_as_binary(    *(data-lowBit/8 ))
    << "\t *(data-lowBit/8-1) " << print_as_binary(    *(data-lowBit/8 -1 ))
    << "\tlowBit:"<< lowBit
    << "\tfirstByteMask :"<<print_as_binary(firstByteMask)
    << "\tsecondByteMask:"<<print_as_binary(secondByteMask)
    << "\tfirstByteBit:"<<print_as_binary(firstByteBit)
    << std::endl;
    }
  */
  return value;
}

/// Get 9 bit words from a bit stream, starting from the right, skipping the first 'skip' bits (0 < skip < 7).
/// Ptr must point to the rightmost byte that has some bits of this word, and is updated by this function
inline uint16_t SiStripPedestals::get10bits(const uint8_t * &ptr, int8_t skip) const {
    uint8_t maskThis = (0xFF << skip);
    uint8_t maskThat = ((4 << skip) - 1);
    uint16_t ret = ( ((*ptr) & maskThis) >> skip );
    --ptr;
    return ret | ( ((*ptr) & maskThat) << (8 - skip) );
}

void
SiStripPedestals::allPeds  (std::vector<int>   & peds,  const Range& range) const {
    size_t mysize  = ((range.second-range.first) << 3) / 10;
    size_t size = peds.size();
    if (mysize < size) throw cms::Exception("CorruptedData") 
            << "[SiStripPedestals::allPeds] Requested pedestals for " << peds.size() << " strips, I have it only for " << mysize << " strips\n";
    size_t size4 = size & (~0x3), carry = size & 0x3; // we have an optimized way of unpacking 4 strips
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(&*range.second) - 1;
    std::vector<int>::iterator out = peds.begin(), end4 = peds.begin() + size4;
    // we do it this baroque way instead of just loopin on all the strips because it's faster
    // as the value of 'skip' is a constant, so the compiler can compute the masks directly
   while (out < end4) {
        *out = static_cast<int> ( get10bits(ptr, 0) ); ++out;
        *out = static_cast<int> ( get10bits(ptr, 2) ); ++out;
        *out = static_cast<int> ( get10bits(ptr, 4) ); ++out;
        *out = static_cast<int> ( get10bits(ptr, 6) ); ++out;
        --ptr; // every 4 strips we have to skip one more bit
    } 
    for (size_t rem = 0; rem < carry; ++rem ) {
        *out = static_cast<int> ( get10bits(ptr, 2*rem) ); ++out;
    }
}

void SiStripPedestals::printSummary(std::stringstream& ss, const TrackerTopology* trackerTopo) const
{
  std::vector<uint32_t> detid;
  getDetIds(detid);
  SiStripDetSummary summary{trackerTopo};
  for( size_t id = 0; id < detid.size(); ++id ) {
    SiStripPedestals::Range range = getRange(detid[id]);
    for( int it=0; it < (range.second-range.first)*8/10; ++it ){
      summary.add( detid[id], getPed(it,range) );
    }
  }
  ss << "Summary of pedestals:" << std::endl;
  summary.print(ss);
}


void SiStripPedestals::printDebug(std::stringstream& ss, const TrackerTopology* /*trackerTopo*/) const
{
  std::vector<uint32_t> detid;
  getDetIds(detid);

  ss << "Number of detids = " << detid.size() << std::endl;

  for( size_t id = 0; id < detid.size(); ++id ) {
    SiStripPedestals::Range range = getRange(detid[id]);

    int strip = 0;
    ss << "detid" << std::setw(15) << "strip" << std::setw(10) << "pedestal" << std::endl;
    int detId = 0;
    int oldDetId = 0;
    for( int it=0; it < (range.second-range.first)*8/10; ++it ){
      detId = detid[id];
      if( detId != oldDetId ) {
        oldDetId = detId;
        ss << detid[id];
      }
      else ss << "   ";
      ss << std::setw(15) << strip++ << std::setw(10) << getPed(it,range) << std::endl;
    }
  }
}


/**
  const std::string SiStripNoises::print_as_binary(const uint8_t ch) const
  {
  std::string     str;
  int i = CHAR_BIT;
  while (i > 0)
  {
  -- i;
  str.push_back((ch&(1 << i) ? '1' : '0'));
  }
  return str;
  }

  std::string SiStripNoises::print_char_as_binary(const unsigned char ch) const
  {
  std::string     str;
  int i = CHAR_BIT;
  while (i > 0)
  {
  -- i;
  str.push_back((ch&(1 << i) ? '1' : '0'));
  }
  return str;
  }

  std::string SiStripNoises::print_short_as_binary(const short ch) const
  {
  std::string     str;
  int i = CHAR_BIT*2;
  while (i > 0)
  {
  -- i;
  str.push_back((ch&(1 << i) ? '1' : '0'));
  }
  return str;
  }
**/
