#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

SiStripPedestals::SiStripPedestals(){}
SiStripPedestals::~SiStripPedestals(){}


std::vector<SiStripPed> SiStripPedestals::getPed(uint32_t DetId) const
{
  std::vector<SiStripPed> _ped;
  SiStripPedestalsMapIterator mapiter=m_pedestals.find(DetId);
  if (mapiter!=m_pedestals.end())
    for (SiStripPedestalsVectorIterator iter=mapiter->second.begin();iter!=mapiter->second.end();iter++)
      _ped.push_back(static_cast<SiStripPed>(( (*iter).StripData >> 22) &0x000003FF)); 
  
  return _ped;
};
std::vector<SiStripNoise> SiStripPedestals::getNoise(uint32_t DetId) const
{
  std::vector<SiStripNoise> _noise;
  SiStripPedestalsMapIterator mapiter=m_pedestals.find(DetId);
  if (mapiter!=m_pedestals.end())
    for (SiStripPedestalsVectorIterator iter=mapiter->second.begin();iter!=mapiter->second.end();iter++)
      _noise.push_back(static_cast<SiStripNoise>(( (*iter).StripData >> 13) &0x000001FF)/10.0);   
  return _noise;
}
std::vector<SiStripDisable>  SiStripPedestals::getDisable(uint32_t DetId) const
{
  std::vector<SiStripDisable> _disable;
  SiStripPedestalsMapIterator mapiter=m_pedestals.find(DetId);
  if (mapiter!=m_pedestals.end())
    for (SiStripPedestalsVectorIterator iter=mapiter->second.begin();iter!=mapiter->second.end();iter++)
      _disable.push_back(static_cast<SiStripDisable>( (*iter).StripData &0x00000001)); 
  return _disable;
}
std::vector<SiStripLowTh>   SiStripPedestals::getLowTh(uint32_t DetId) const
{
  std::vector<SiStripLowTh> _lowth;
  SiStripPedestalsMapIterator mapiter=m_pedestals.find(DetId);
  if (mapiter!=m_pedestals.end())
    for (SiStripPedestalsVectorIterator iter=mapiter->second.begin();iter!=mapiter->second.end();iter++)
      _lowth.push_back(static_cast<SiStripLowTh>(( (*iter).StripData >> 1) &0x0000003F)/5.0);
  return _lowth;
}
std::vector<SiStripHighTh>   SiStripPedestals::getHighTh(uint32_t DetId) const
{
  std::vector<SiStripHighTh> _highth;
  SiStripPedestalsMapIterator mapiter=m_pedestals.find(DetId);
  if (mapiter!=m_pedestals.end())
    for (SiStripPedestalsVectorIterator iter=mapiter->second.begin();iter!=mapiter->second.end();iter++)
      _highth.push_back(static_cast<SiStripHighTh>(( (*iter).StripData >> 7) &0x0000003F)/5.0);
  return _highth;
}

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
