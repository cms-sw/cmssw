#ifndef mySiStripNoises_h
#define mySiStripNoises_h

#include<vector>
//#include<map>
//#include<iostream>
//#include<boost/cstdint.hpp>


//typedef float SiStripNoise;
//typedef bool  SiStripDisable;

class mySiStripNoises {

 public:
  mySiStripNoises(){}
  ~mySiStripNoises(){}

  struct SiStripData {
    float   getNoise()   const; 
    bool getDisable() const;
    void setData(short data);
    void setData(float noisevalue,bool disable);
    short Data; // Data = sign(+/-1) * Noise(Adc count). if Data <=0 then strip is disable
  };
  
  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  };
  
  typedef std::vector<short>                               SiStripNoiseVector;
  typedef SiStripNoiseVector::const_iterator               ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
  
  bool put(const uint32_t& detID,Range input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds) const;

 private:
  //SiStripNoiseVector v_noises;
  //Registry indexes;
  std::vector<short>  v_noises;
  std::vector<DetRegistry> indexes;
};

class StrictWeakOrdering{
  public:
  bool operator() (const mySiStripNoises::DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
};

#endif
