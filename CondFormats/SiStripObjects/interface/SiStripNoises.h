#ifndef SiStripNoises_h
#define SiStripNoises_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class SiStripNoises {

 public:

  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  };

  class StrictWeakOrdering{
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
  SiStripNoises(){};
  ~SiStripNoises(){};

  bool put(const uint32_t& detID,const InputVector &input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;
  
  ContainerIterator getDataVectorBegin()    const {return v_noises.begin();}
  ContainerIterator getDataVectorEnd()      const {return v_noises.end();}
  RegistryIterator getRegistryVectorBegin() const {return indexes.begin();}
  RegistryIterator getRegistryVectorEnd()   const{return indexes.end();}

  float   getNoise  (const uint16_t& strip, const Range& range) const;
  void    allNoises (std::vector<float> & noises, const Range& range) const;
  void    setData(float noise_, InputVector& vped);

  void printDebug(std::stringstream& ss) const;
  void printSummary(std::stringstream& ss) const;

 private:
  void     encode(const InputVector& Vi, std::vector<unsigned char>& Vo_CHAR);
  uint16_t decode (const uint16_t& strip, const Range& range) const;
  /// Get 9 bits from a bit stream, starting from the right, skipping the first 'skip' bits (0 < skip < 8).
  /// Ptr must point to the rightmost bit, and is updated by this function
  inline uint16_t get9bits(const uint8_t * &ptr, int8_t skip) const ;

  Container 	v_noises; 
  Registry 	indexes;

  /*
    const std::string print_as_binary(const uint8_t ch) const;
    std::string print_char_as_binary(const unsigned char ch) const;
    std::string print_short_as_binary(const short ch) const;
  */
};

#endif
