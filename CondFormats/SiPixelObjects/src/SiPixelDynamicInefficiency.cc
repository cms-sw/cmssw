#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiPixelDynamicInefficiency::SiPixelDynamicInefficiency(){theInstLumiScaleFactor_=-9999;}

bool SiPixelDynamicInefficiency::putPixelGeomFactor (const uint32_t& detid, double& value){
  std::map<unsigned int,double>::const_iterator id=m_PixelGeomFactors.find(detid);
  if(id!=m_PixelGeomFactors.end()){
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency PixelGeomFactor for DetID " << detid << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  else m_PixelGeomFactors[detid]=value;
  return true;
}

double SiPixelDynamicInefficiency::getPixelGeomFactor (const uint32_t& detid) const  {
  std::map<unsigned int,double>::const_iterator id=m_PixelGeomFactors.find(detid);
  if(id!=m_PixelGeomFactors.end()) return id->second;
  else {
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency PixelGeomFactor for DetID " << detid << " is not stored" << std::endl; 
  } 
  return 0;
}

bool SiPixelDynamicInefficiency::putColGeomFactor (const uint32_t& detid, double& value){
  std::map<unsigned int,double>::const_iterator id=m_ColGeomFactors.find(detid);
  if(id!=m_ColGeomFactors.end()){
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency ColGeomFactor for DetID " << detid << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  else m_ColGeomFactors[detid]=value;
  return true;
}

double SiPixelDynamicInefficiency::getColGeomFactor (const uint32_t& detid) const  {
  std::map<unsigned int,double>::const_iterator id=m_ColGeomFactors.find(detid);
  if(id!=m_ColGeomFactors.end()) return id->second;
  else {
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency ColGeomFactor for DetID " << detid << " is not stored" << std::endl; 
  } 
  return 0;
}

bool SiPixelDynamicInefficiency::putChipGeomFactor (const uint32_t& detid, double& value){
  std::map<unsigned int,double>::const_iterator id=m_ChipGeomFactors.find(detid);
  if(id!=m_ChipGeomFactors.end()){
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency ChipGeomFactor for DetID " << detid << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  else m_ChipGeomFactors[detid]=value;
  return true;
}

double SiPixelDynamicInefficiency::getChipGeomFactor (const uint32_t& detid) const  {
  std::map<unsigned int,double>::const_iterator id=m_ChipGeomFactors.find(detid);
  if(id!=m_ChipGeomFactors.end()) return id->second;
  else {
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency ChipGeomFactor for DetID " << detid << " is not stored" << std::endl; 
  } 
  return 0;
}

bool SiPixelDynamicInefficiency::putPUFactor (const uint32_t& detid, std::vector<double>& v_value){
  std::map<unsigned int,std::vector<double> >::const_iterator id=m_PUFactors.find(detid);
  if(id!=m_PUFactors.end()){
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency PUFactor for DetID " << detid << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  else m_PUFactors[detid]=v_value;
  return true;
}

std::vector<double> SiPixelDynamicInefficiency::getPUFactor (const uint32_t& detid) const {
  std::map<unsigned int,std::vector<double> >::const_iterator id=m_PUFactors.find(detid);
  if(id!=m_PUFactors.end()) return id->second;
  else {
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency PUFactor for DetID " << detid << " is not stored" << std::endl; 
  } 
  std::vector<double> empty;
  return empty;
}

bool SiPixelDynamicInefficiency::putDetIdmask(uint32_t& mask){
  for (unsigned int i=0;i<v_DetIdmasks.size();i++) if (mask == v_DetIdmasks.at(i)) return false;
  v_DetIdmasks.push_back(mask);
  return true;
}
uint32_t SiPixelDynamicInefficiency::getDetIdmask(unsigned int& i) const{
  if (v_DetIdmasks.size() <= i) {
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency DetIdmask "<<i<<" is not stored!" << std::endl;
    return 0;
  }
  else return v_DetIdmasks.at(i);
}

bool SiPixelDynamicInefficiency::puttheInstLumiScaleFactor(double& theInstLumiScaleFactor){
  if (theInstLumiScaleFactor_ != -9999){
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency theInstLumiScaleFactor is already stored! Skippig this put!" << std::endl;
    return false;
  }
  else {
    theInstLumiScaleFactor_ = theInstLumiScaleFactor;
    return true;
  }
}

double SiPixelDynamicInefficiency::gettheInstLumiScaleFactor() const {
  if (theInstLumiScaleFactor_ == -9999) {
    edm::LogError("SiPixelDynamicInefficiency") << "SiPixelDynamicInefficiency theInstLumiScaleFactor is not stored!" << std::endl;
    return 0;
  }
  else return theInstLumiScaleFactor_;
}
