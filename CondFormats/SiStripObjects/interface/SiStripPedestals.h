#ifndef SiStripPedestals_h
#define SiStripPedestals_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


typedef int   SiStripPed;
typedef float SiStripNoise;
typedef float SiStripLowTh;
typedef float SiStripHighTh;
typedef bool  SiStripDisable;



class SiStripPedestals {

 public:
  SiStripPedestals();
  ~SiStripPedestals();
    
  std::vector<SiStripPed>     getPed(uint32_t DetId) const;
  std::vector<SiStripNoise>   getNoise(uint32_t DetId) const;
  std::vector<SiStripDisable> getDisable(uint32_t DetId) const;
  std::vector<SiStripLowTh>   getLowTh(uint32_t DetId) const;
  std::vector<SiStripHighTh>  getHighTh(uint32_t DetId) const;
   
  uint32_t EncodeStripData(float ped_, float noise_, float lowTh_, float highTh_, bool disable_); 
  uint32_t EncodeStripData(float ped_, float noise_, float lowTh_, float highTh_, bool disable_,bool debug); 
    
  struct Item {
    uint32_t StripData;
  };

  typedef std::vector<Item>                 SiStripPedestalsVector;
  typedef std::vector<Item>::const_iterator SiStripPedestalsVectorIterator;
  
  typedef std::map<uint32_t, SiStripPedestalsVector>                 SiStripPedestalsMap;
  typedef std::map<uint32_t, SiStripPedestalsVector>::const_iterator SiStripPedestalsMapIterator;
  
  std::map<uint32_t, SiStripPedestalsVector> m_pedestals;
};

#endif
