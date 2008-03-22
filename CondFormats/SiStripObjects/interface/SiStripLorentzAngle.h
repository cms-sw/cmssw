#ifndef SiStripLorentzAngle_h
#define SiStripLorentzAngle_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class SiStripLorentzAngle {

 public:
 
  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
  };

  SiStripLorentzAngle(){};
  ~SiStripLorentzAngle(){};

  inline void putLorentsAngles(std::map<unsigned int,float>& LA){m_LA=LA;}   
  inline const std::map<unsigned int,float>&  getLorentzAngles () const {return m_LA;}

  bool   putLorentzAngle(const uint32_t&, float&);
  const float&  getLorentzAngle (const uint32_t&) const;

  ContainerIterator getDataVectorBegin()    const {return v_LA.begin();}
  ContainerIterator getDataVectorEnd()      const {return v_LA.end();}
  RegistryIterator getRegistryVectorBegin() const {return indexes.begin();}
  RegistryIterator getRegistryVectorEnd()   const{return indexes.end();}


 private:
  std::map<unsigned int,float> m_LA; 
  std::vector<float> v_LA; 
  std::vector<DetRegistry> indexes;
};

#endif
