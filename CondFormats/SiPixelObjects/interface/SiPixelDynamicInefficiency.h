#ifndef SiPixelDynamicInefficiency_h
#define SiPixelDynamicInefficiency_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>


class SiPixelDynamicInefficiency {

 public:
 
  SiPixelDynamicInefficiency();
  ~SiPixelDynamicInefficiency(){};

  inline void putPixelGeomFactors (std::map<unsigned int,double>& PixelGeomFactors){m_PixelGeomFactors=PixelGeomFactors;} 
  inline const std::map<unsigned int,double>&  getPixelGeomFactors () const {return m_PixelGeomFactors;}

  inline void putColGeomFactors (std::map<unsigned int,double>& ColGeomFactors){m_ColGeomFactors=ColGeomFactors;} 
  inline const std::map<unsigned int,double>&  getColGeomFactors () const {return m_ColGeomFactors;}

  inline void putChipGeomFactors (std::map<unsigned int,double>& ChipGeomFactors){m_ChipGeomFactors=ChipGeomFactors;} 
  inline const std::map<unsigned int,double>&  getChipGeomFactors () const {return m_ChipGeomFactors;}

  inline void putPUFactors (std::map<unsigned int,std::vector<double> >& PUFactors){m_PUFactors=PUFactors;} 
  inline const std::map<unsigned int,std::vector<double> >&  getPUFactors () const {return m_PUFactors;}

  inline void puttheInstLumiScaleFactor_(double& InstLumiScaleFactor){theInstLumiScaleFactor_=InstLumiScaleFactor;}
  inline const double gettheInstLumiScaleFactor_() const {return theInstLumiScaleFactor_;}

  inline void putDetIdmasks(std::vector<uint32_t>& masks){v_DetIdmasks=masks;}
  inline const std::vector<uint32_t> getDetIdmasks() const {return v_DetIdmasks;}

  bool   putPixelGeomFactor (const uint32_t&, double&);
  double  getPixelGeomFactor (const uint32_t&) const;

  bool   putColGeomFactor (const uint32_t&, double&);
  double  getColGeomFactor (const uint32_t&) const;

  bool   putChipGeomFactor (const uint32_t&, double&);
  double  getChipGeomFactor (const uint32_t&) const;

  bool putPUFactor (const uint32_t&, std::vector<double>&);
  std::vector<double> getPUFactor (const uint32_t&) const;

  bool putDetIdmask(uint32_t&);
  uint32_t getDetIdmask(unsigned int&) const;

  bool puttheInstLumiScaleFactor(double&);
  double gettheInstLumiScaleFactor() const;

 private:
  std::map<unsigned int,double> m_PixelGeomFactors;
  std::map<unsigned int,double> m_ColGeomFactors;
  std::map<unsigned int,double> m_ChipGeomFactors;
  std::map<unsigned int,std::vector<double> > m_PUFactors;
  std::vector<uint32_t> v_DetIdmasks;
  double theInstLumiScaleFactor_;

 COND_SERIALIZABLE;
};

#endif
