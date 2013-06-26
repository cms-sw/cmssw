#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

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

float SiStripNoises::getNoise(uint16_t strip, const Range& range) {
	if (9*strip>=(range.second-range.first)*8){
		throw cms::Exception("CorruptedData")
			<< "[SiStripNoises::getNoise] looking for SiStripNoises for a strip out of range: strip " << strip;
	}
	return getNoiseFast(strip,range);
}

void SiStripNoises::setData(float noise_, InputVector& v){
	v.push_back((static_cast<int16_t>  (noise_*10.0 + 0.5) & 0x01FF)) ;
}

void SiStripNoises::encode(const InputVector& Vi, std::vector<unsigned char>& Vo){
  static const uint16_t  BITS_PER_STRIP  = 9;
  const size_t           VoSize          = (size_t)((Vi.size() *       BITS_PER_STRIP)/8+.999);
  Vo.resize(VoSize);
  for(size_t i = 0; i<VoSize; ++i)
    Vo[i]   &=      0x00u;
  
  for(unsigned int stripIndex =0; stripIndex<Vi.size(); ++stripIndex){
    unsigned char*  data    =       &Vo[VoSize-1];
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


//============ Methods for bulk-decoding all noises for a module ================



void SiStripNoises::allNoises(std::vector<float> &noises, const Range& range) const {
    size_t mysize  = ((range.second-range.first) << 3) / 9;
    size_t size = noises.size();
    if (mysize < size) throw cms::Exception("CorruptedData") 
            << "[SiStripNoises::allNoises] Requested noise for " << noises.size() << " strips, I have it only for " << mysize << " strips\n";
    size_t size8 = size & (~0x7), carry = size & 0x7; // we have an optimized way of unpacking 8 strips
    const uint8_t *ptr = (&*range.second) - 1;
    std::vector<float>::iterator out = noises.begin(), end8 = noises.begin() + size8;
    // we do it this baroque way instead of just loopin on all the strips because it's faster
    // as the value of 'skip' is a constant, so the compiler can compute the masks directly
   while (out < end8) {
        *out = static_cast<float> ( get9bits(ptr, 0) / 10.0f ); ++out;
        *out = static_cast<float> ( get9bits(ptr, 1) / 10.0f ); ++out;
        *out = static_cast<float> ( get9bits(ptr, 2) / 10.0f ); ++out;
        *out = static_cast<float> ( get9bits(ptr, 3) / 10.0f ); ++out;
        *out = static_cast<float> ( get9bits(ptr, 4) / 10.0f ); ++out;
        *out = static_cast<float> ( get9bits(ptr, 5) / 10.0f ); ++out;
        *out = static_cast<float> ( get9bits(ptr, 6) / 10.0f ); ++out;
        *out = static_cast<float> ( get9bits(ptr, 7) / 10.0f ); ++out;
        --ptr; // every 8 strips we have to skip one more bit
    } 
    for (size_t rem = 0; rem < carry; ++rem ) {
        *out = static_cast<float> ( get9bits(ptr, rem) / 10.0f ); ++out;
    }
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

void SiStripNoises::printDebug(std::stringstream& ss) const{
  RegistryIterator rit=getRegistryVectorBegin(), erit=getRegistryVectorEnd();
  uint16_t Nstrips;
  std::vector<float> vstripnoise;

  ss << "detid" << std::setw(15) << "strip" << std::setw(10) << "noise" << std::endl;

  int detId = 0;
  int oldDetId = 0;
  for(;rit!=erit;++rit){
    Nstrips = (rit->iend-rit->ibegin)*8/9; //number of strips = number of chars * char size / strip noise size
    vstripnoise.resize(Nstrips);
    allNoises(vstripnoise,make_pair(getDataVectorBegin()+rit->ibegin,getDataVectorBegin()+rit->iend));

    detId = rit->detid;
    if( detId != oldDetId ) {
      oldDetId = detId;
      ss << detId;
    }
    else ss << "   ";
    for(size_t i=0;i<Nstrips;++i){
      if( i != 0 ) ss << "   ";
      ss << std::setw(15) << i << std::setw(10) << vstripnoise[i] << std::endl;
    }
  }
}

void SiStripNoises::printSummary(std::stringstream& ss) const{

  SiStripDetSummary summary;

  std::stringstream tempss;

  RegistryIterator rit=getRegistryVectorBegin(), erit=getRegistryVectorEnd();
  uint16_t Nstrips;
  std::vector<float> vstripnoise;
  double mean,rms,min, max;
  for(;rit!=erit;++rit){
    Nstrips = (rit->iend-rit->ibegin)*8/9; //number of strips = number of chars * char size / strip noise size
    vstripnoise.resize(Nstrips);
    allNoises(vstripnoise,make_pair(getDataVectorBegin()+rit->ibegin,getDataVectorBegin()+rit->iend));
    tempss << "\ndetid: " << rit->detid << " \t ";
    mean=0; rms=0; min=10000; max=0;  

    DetId detId(rit->detid);

    for(size_t i=0;i<Nstrips;++i){
      mean+=vstripnoise[i];
      rms+=vstripnoise[i]*vstripnoise[i];
      if(vstripnoise[i]<min) min=vstripnoise[i];
      if(vstripnoise[i]>max) max=vstripnoise[i];

      summary.add(detId, vstripnoise[i]);
    }
    mean/=Nstrips;
    rms= sqrt(rms/Nstrips-mean*mean);


    tempss << "Nstrips " << Nstrips << " \t; mean " << mean << " \t; rms " << rms << " \t; min " << min << " \t; max " << max << "\t " ; 
  }
  ss << std::endl << "Summary:" << std::endl;
  summary.print(ss);
  ss << std::endl;
  ss << tempss.str();
}

std::vector<SiStripNoises::ratioData> SiStripNoises::operator / ( const SiStripNoises& d) {
  std::vector<ratioData> result;
  ratioData aData;

  RegistryIterator iter=getRegistryVectorBegin();
  RegistryIterator iterE=getRegistryVectorEnd();

  //Divide result by d
  for(;iter!=iterE;++iter){
    float value;
    //get noise from d
    aData.detid=iter->detid;
    aData.values.clear();
    Range d_range=d.getRange(iter->detid);
    Range range=Range(v_noises.begin()+iter->ibegin,v_noises.begin()+iter->iend);

    //if denominator is missing, put the ratio value to 0xFFFF (=inf)
    size_t strip=0, stripE= (range.second-range.first)*8/9;
    for (;strip<stripE;++strip){       
      if(d_range.first==d_range.second){
	value=0xFFFF;
      }else{
	value=getNoise(strip,range)/d.getNoise(strip,d_range);
      }
      aData.values.push_back(value);
    }
    result.push_back(aData);
  }

  iter=d.getRegistryVectorBegin();
  iterE=d.getRegistryVectorEnd();

  //Divide result by d
  for(;iter!=iterE;++iter){
    float value;
    //get noise from d
    Range range=this->getRange(iter->detid);
    Range d_range=Range(d.v_noises.begin()+iter->ibegin,d.v_noises.begin()+iter->iend);
    if(range.first==range.second){
      aData.detid=iter->detid;
      aData.values.clear();
      size_t strip=0, stripE= (d_range.second-d_range.first)*8/9;
      for (;strip<stripE;++strip){       
	value=0.;
	aData.values.push_back(value);
      }
      result.push_back(aData);
    }
  }
  
  return result;
}
