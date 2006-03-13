#ifndef SiStripNoises_h
#define SiStripNoises_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


typedef float SiStripNoise;
typedef bool  SiStripDisable;

class SiStripNoises {

 public:

  class SiStripData {
  public:
  
    SiStripNoise   getNoise()   const {return static_cast<SiStripNoise> (abs(Data)/10.0);}
    SiStripDisable getDisable() const {return ( (Data>0) ? false : true );}
    void setData(int16_t data){Data=data;}
    void setData(float noise_,bool disable_){
      int16_t noise =  static_cast<int16_t>  (noise_*10.0 + 0.5) & 0x01FF;
      Data = ( disable_ ? -1 : 1 ) * noise;
    };
    
  private:
    int16_t Data; // Data = sign(+/-1) * Noise(Adc count). if Data <=0 then strip is disable
  };
  
  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  };

  typedef std::vector<SiStripData>                         SiStripNoiseVector;
  typedef SiStripNoiseVector::const_iterator               ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
 
  SiStripNoises(){};
  ~SiStripNoises(){};
    
  bool put(const uint32_t& detID,Range input);
  const Range getRange(const uint32_t& detID) const;
  void getDetIds(std::vector<uint32_t>& DetIds_) const;

 private:
  SiStripNoiseVector v_noises;
  Registry indexes;

};

class StrictWeakOrdering{
  public:
  bool operator() (const SiStripNoises::DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
};

#endif
