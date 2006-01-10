#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

SiStripPedestals::SiStripPedestals(){}
SiStripPedestals::~SiStripPedestals(){}

const std::vector<SiStripPedestals::SiStripData> & SiStripPedestals::getSiStripPedestalsVector(const uint32_t & DetId) const {
  SiStripPedestalsMapIterator mapiter=m_pedestals.find(DetId);
  if (mapiter!=m_pedestals.end())
    return mapiter->second;
  return SiStripPedestalsVector();
};


uint32_t SiStripPedestals::EncodeStripData(float ped_, float noise_, float lowTh_, float highTh_, bool disable_)
{
  // Encoding Algorithm from Fed9UUtils/src/Fed9UDescriptionToXml.cc
  
  uint32_t low   = (static_cast<uint32_t>(lowTh_*5.0 + 0.5)  ) & 0x3F; 
  uint32_t high  = (static_cast<uint32_t>(highTh_*5.0 + 0.5) ) & 0x3F;
  uint32_t noise =  static_cast<uint32_t>(noise_*10.0 + 0.5)   & 0x01FF;
  uint32_t ped   =  static_cast<uint32_t>(ped_)                & 0x03FF;
  
  uint32_t stripData = (ped << 22) | (noise << 13) | (high << 7) | (low << 1) | ( disable_ ? 0x1 : 0x0 );
  
  return stripData;
};


uint32_t SiStripPedestals::EncodeStripData(float ped_, float noise_, float lowTh_, float highTh_, bool disable_, bool debug)
{
  // Encoding Algorithm from Fed9UUtils/src/Fed9UDescriptionToXml.cc
  
  uint32_t low   = (static_cast<uint32_t>(lowTh_*5.0 + 0.5)  ) & 0x3F; 
  uint32_t high  = (static_cast<uint32_t>(highTh_*5.0 + 0.5) ) & 0x3F;
  uint32_t noise =  static_cast<uint32_t>(noise_*10.0 + 0.5)   & 0x01FF;
  uint32_t ped   =  static_cast<uint32_t>(ped_)                & 0x03FF;
  
  uint32_t stripData = (ped << 22) | (noise << 13) | (high << 7) | (low << 1) | ( disable_ ? 0x1 : 0x0 );


  if (debug)
    std::cout 
      << std::fixed << ped_       << " \t" 
      << std::fixed << noise_     << " \t" 
      << lowTh_     << " \t" 
      << highTh_    << " \t" 
      << disable_  << " \t" 
      << stripData << " \t" 
      << std::endl;
  
  return stripData;
};
