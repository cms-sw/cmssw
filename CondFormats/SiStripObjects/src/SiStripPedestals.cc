#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "FWCore/Utilities/interface/Exception.h"

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


void SiStripPedestals::setData(float ped, std::vector<short>& vped){

	//unsigned int ped_  = (static_cast<unsigned int>(ped)) & 0xFFF; 
	//vped.resize(vped.size()+3);
	// insert in vector of char
	//::memcpy((void*)(&vped[vped.size()-3]),(void*)(&data),3);
	vped.push_back((static_cast<int16_t>  (ped) & 0x3FF)) ;
}

float SiStripPedestals::getPed(const uint16_t& strip, const Range& range) const {
	if (10*strip>=(range.second-range.first)*8){
		throw cms::Exception("CorruptedData")
			<< "[SiStripPedestals::getPed] looking for SiStripPedestals for a strip out of range: strip " << strip;
	}
	//const DecodingStructure & s = (const DecodingStructure & ) *(range.first+strip*3);
	//return (s.ped & 0x3FF);
	return   static_cast<float> (decode(strip,range));
}

void SiStripPedestals::encode(InputVector& Vi, std::vector<unsigned char>& Vo){
  static const uint16_t  BITS_PER_STRIP  = 10;
  const size_t           VoSize          = (size_t)((Vi.size() *       BITS_PER_STRIP)/8+.999);
  Vo.resize(VoSize);
  for(size_t i = 0; i<Vo.size(); ++i)
    Vo[i]   &=      0x00u;
  
  for(int stripIndex =0; stripIndex<Vi.size(); ++stripIndex){
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
