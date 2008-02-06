#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiStripLorentzAngle::putLorentzAngle(const uint32_t& detid, float& value){
  std::map<unsigned int,float>::const_iterator id=m_LA.find(detid);
  if(id!=m_LA.end()){
    edm::LogError("SiStripLorentzAngle") << "SiStripLorentzAngle for DetID " << detid << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  else m_LA[detid]=value;
  return true;
}
const float& SiStripLorentzAngle::getLorentzAngle(const uint32_t& detid) const  {
  std::map<unsigned int,float>::const_iterator id=m_LA.find(detid);
  if(id!=m_LA.end()) return id->second;
  else {
    edm::LogError("SiStripLorentzAngle") << "SiStripLorentzAngle for DetID " << detid << " is not stored" << std::endl; 
  }
  static float temp = 0.; // added by R.B. 
  return temp;
}
