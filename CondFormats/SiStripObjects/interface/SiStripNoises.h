#ifndef SiStripNoises_h
#define SiStripNoises_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

/**
 * Stores the noise value for all the strips. <br>
 * The values are encoded from a vector<uint16_t> to a vector<unsigned char> <br>
 *
 * The printSummary method prints: Nstrips, mean, rms, min and max noise for each detId.
 * The print Debug method prints the noise for every strip.
 *
 */

class SiStripNoises
{
 public:

  struct ratioData{
    uint32_t detid;
    std::vector<float> values;
  };
  
  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  };

  class StrictWeakOrdering
  {
   public:
    bool operator() (const DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
  };

  typedef std::vector<unsigned char>                       Container;  
  typedef std::vector<unsigned char>::const_iterator       ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
  typedef std::vector<uint16_t>          		   InputVector;

  SiStripNoises(const SiStripNoises& );
  SiStripNoises(){}
  ~SiStripNoises(){}

  bool put(const uint32_t& detID,const InputVector &input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;
  
  ContainerIterator getDataVectorBegin()    const {return v_noises.begin();}
  ContainerIterator getDataVectorEnd()      const {return v_noises.end();}
  RegistryIterator getRegistryVectorBegin() const {return indexes.begin();}
  RegistryIterator getRegistryVectorEnd()   const{return indexes.end();}

  static inline float getNoiseFast(const uint16_t& strip, const Range& range) {
    return  0.1f*float(decode(strip,range));
  }

  static float getNoise(uint16_t strip, const Range& range);

  void    allNoises (std::vector<float> & noises, const Range& range) const;
  void    setData(float noise_, InputVector& vped);

  void printDebug(std::stringstream& ss) const;
  void printSummary(std::stringstream& ss) const;

  std::vector<ratioData> operator / (const SiStripNoises& d) ;

 private:
  static  void encode(const InputVector& Vi, std::vector<unsigned char>& Vo_CHAR);

  static inline uint16_t decode (uint16_t strip, const Range& range);

  /// Get 9 bits from a bit stream, starting from the right, skipping the first 'skip' bits (0 < skip < 8).
  /// Ptr must point to the rightmost bit, and is updated by this function
  static inline uint16_t get9bits(const uint8_t * &ptr, int8_t skip);

  Container 	v_noises; 
  Registry 	indexes;


  /*
    const std::string print_as_binary(const uint8_t ch) const;
    std::string print_char_as_binary(const unsigned char ch) const;
    std::string print_short_as_binary(const short ch) const;
  */
};

/// Get 9 bit words from a bit stream, starting from the right, skipping the first 'skip' bits (0 < skip < 8).
/// Ptr must point to the rightmost byte that has some bits of this word, and is updated by this function
inline uint16_t SiStripNoises::get9bits(const uint8_t * &ptr, int8_t skip) {
    uint8_t maskThis = (0xFF << skip);
    uint8_t maskThat = ((2 << skip) - 1);
    uint16_t ret = ( ((*ptr) & maskThis) >> skip );
    --ptr;
    return ret | ( ((*ptr) & maskThat) << (8 - skip) );
}

inline uint16_t SiStripNoises::decode (uint16_t strip, const Range& range) {
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


#endif
