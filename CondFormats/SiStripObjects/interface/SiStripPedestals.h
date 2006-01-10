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
  
/*   struct Item { */
/*     uint32_t StripData; */
/*   }; */
  
  class SiStripData {
  public:

    SiStripPed     getPed()     const {return static_cast<SiStripPed>    ((Data >> 22) &0x000003FF);     } 
    SiStripNoise   getNoise()   const {return static_cast<SiStripNoise>  ((Data >> 13) &0x000001FF)/10.0;}
    SiStripDisable getDisable() const {return static_cast<SiStripDisable>(Data &0x00000001);             }
    SiStripLowTh   getLowTh()   const {return static_cast<SiStripLowTh>  ((Data >> 1) &0x0000003F)/5.0;  }
    SiStripHighTh  getHighTh()  const {return static_cast<SiStripHighTh> ((Data >> 7) &0x0000003F)/5.0;  }

    uint32_t Data;
  };
  


    
  uint32_t EncodeStripData(float ped_, float noise_, float lowTh_, float highTh_, bool disable_); 
  uint32_t EncodeStripData(float ped_, float noise_, float lowTh_, float highTh_, bool disable_,bool debug); 


  const std::vector<SiStripData> &  getSiStripPedestalsVector(const uint32_t & DetId) const;

  std::map<uint32_t, std::vector<SiStripData> > m_pedestals;
};

typedef std::vector<SiStripPedestals::SiStripData>                 SiStripPedestalsVector;
typedef std::vector<SiStripPedestals::SiStripData>::const_iterator SiStripPedestalsVectorIterator;
typedef std::map<uint32_t, SiStripPedestalsVector>                 SiStripPedestalsMap;
typedef std::map<uint32_t, SiStripPedestalsVector>::const_iterator SiStripPedestalsMapIterator;


#endif
