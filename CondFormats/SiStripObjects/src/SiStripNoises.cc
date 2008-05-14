#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/Utilities/interface/Exception.h"

SiStripNoises::SiStripNoises(const SiStripNoises& input){
  v_noises.clear();
  indexes.clear();
  v_noises.insert(v_noises.end(),input.v_noises.begin(),input.v_noises.end());
  indexes.insert(indexes.end(),input.indexes.begin(),input.indexes.end());
}

bool SiStripNoises::put(const uint32_t& DetId, const InputVector& input) {
	std::vector<unsigned char>	Vo_CHAR;
	encode(input, Vo_CHAR);

	Registry::iterator p 	= 	std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripNoises::StrictWeakOrdering());
	if (p!=indexes.end() 	&& 	p->detid==DetId)
	  return false;

	size_t sd = Vo_CHAR.end() - Vo_CHAR.begin();
	DetRegistry detregistry;
	detregistry.detid  = DetId;
	detregistry.ibegin = v_noises.size();
	detregistry.iend   = v_noises.size()+sd;
	indexes.insert(p,detregistry);
	v_noises.insert(v_noises.end(),Vo_CHAR.begin(),Vo_CHAR.end());
	return true;
}

const SiStripNoises::Range SiStripNoises::getRange(const uint32_t& DetId) const {
	// get SiStripNoises Range of DetId

	RegistryIterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripNoises::StrictWeakOrdering());
	if (p==indexes.end()|| p->detid!=DetId) 
		return SiStripNoises::Range(v_noises.end(),v_noises.end()); 
	else 
		return SiStripNoises::Range(v_noises.begin()+p->ibegin,v_noises.begin()+p->iend);
}

void SiStripNoises::getDetIds(std::vector<uint32_t>& DetIds_) const {
	// returns vector of DetIds in map
	SiStripNoises::RegistryIterator begin = indexes.begin();
	SiStripNoises::RegistryIterator end   = indexes.end();
	for (SiStripNoises::RegistryIterator p=begin; p != end; ++p) {
		DetIds_.push_back(p->detid);
	}
}

float SiStripNoises::getNoise(const uint16_t& strip, const Range& range) const {
	if (9*strip>=(range.second-range.first)*8){
		throw cms::Exception("CorruptedData")
			<< "[SiStripNoises::getNoise] looking for SiStripNoises for a strip out of range: strip " << strip;
	}
	return   static_cast<float> (decode(strip,range)/10.0);
}

void SiStripNoises::setData(float noise_, InputVector& v){
	v.push_back((static_cast<int16_t>  (noise_*10.0 + 0.5) & 0x01FF)) ;
}

void SiStripNoises::encode(const InputVector& Vi, std::vector<unsigned char>& Vo){
  static const uint16_t  BITS_PER_STRIP  = 9;
  const size_t           VoSize          = (size_t)((Vi.size() *       BITS_PER_STRIP)/8+.999);
  Vo.resize(VoSize);
  for(size_t i = 0; i<Vo.size(); ++i)
    Vo[i]   &=      0x00u;
  
  for(int stripIndex =0; stripIndex<Vi.size(); ++stripIndex){
    unsigned char*  data    =       &Vo[Vo.size()-1];
    uint32_t lowBit         =       stripIndex * BITS_PER_STRIP;
    uint8_t firstByteBit    =       (lowBit & 0x7);
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

uint16_t SiStripNoises::decode (const uint16_t& strip, const Range& range) const{
  const unsigned char *data = &*(range.second -1);  // pointer to the last byte of data
  static const uint16_t BITS_PER_STRIP = 9;

  uint32_t lowBit        = strip * BITS_PER_STRIP;
  uint8_t firstByteBit   = (lowBit & 7);//module 8
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

/*
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
*/
