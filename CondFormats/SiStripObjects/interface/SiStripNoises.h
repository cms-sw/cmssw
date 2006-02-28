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
  SiStripNoises();
  ~SiStripNoises();
  
/*   struct Item { */
/*     uint32_t StripData; */
/*   }; */
  
  class SiStripData {
  public:

    SiStripNoise   getNoise()   const {return static_cast<SiStripNoise> (abs(Data)/10.0);}
    SiStripDisable getDisable() const {return ( (Data>0) ? false : true );}
    void setData(short data){Data=data;}
    void setData(float noise_,bool disable_){
      short noise =  static_cast<short>  (noise_*10.0 + 0.5) & 0x01FF;
      Data = ( disable_ ? -1 : 1 ) * noise;
/*       std::cout  */
/* 	<< std::fixed << noise_ << " \t"  */
/* 	<< disable_  << " \t"  */
/* 	<< Data << " \t"  */
/* 	<< std::endl; */
    };
    
  private:
    //FIXME 
    //the short type is assured to be 16 bit in CMSSW???
    short Data; // Data = sign(+/-1) * Noise(Adc count). if Data <=0 then strip is disable
  };
    
  const std::vector<SiStripData> &  getSiStripNoiseVector(const uint32_t & DetId) const;

  std::map<uint32_t, std::vector<SiStripData> > m_noises;
};

typedef std::vector<SiStripNoises::SiStripData>                 SiStripNoiseVector;
typedef std::vector<SiStripNoises::SiStripData>::const_iterator SiStripNoiseVectorIterator;
typedef std::map<uint32_t, SiStripNoiseVector>                 SiStripNoiseMap;
typedef std::map<uint32_t, SiStripNoiseVector>::const_iterator SiStripNoiseMapIterator;

#endif
