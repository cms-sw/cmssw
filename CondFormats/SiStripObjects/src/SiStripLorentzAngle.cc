#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 

bool SiStripLorentzAngle::putLorentzAngle(const uint32_t& detid, float value){
  std::map<unsigned int,float>::const_iterator id=m_LA.find(detid);
  if(id!=m_LA.end()){
    edm::LogError("SiStripLorentzAngle") << "SiStripLorentzAngle for DetID " << detid << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  else m_LA[detid]=value;
  return true;
}
float SiStripLorentzAngle::getLorentzAngle(const uint32_t& detid) const  {
  std::map<unsigned int,float>::const_iterator id=m_LA.find(detid);
  if(id!=m_LA.end()) return id->second;
  else {
    edm::LogError("SiStripLorentzAngle") << "SiStripLorentzAngle for DetID " << detid << " is not stored" << std::endl; 
  }
  return 0;
}

void SiStripLorentzAngle::printDebug(std::stringstream& ss) const
{
  std::map<unsigned int,float> detid_la = getLorentzAngles();
  std::map<unsigned int,float>::const_iterator it;
  size_t count=0;
  ss << "SiStripLorentzAngleReader:" << std::endl;
  ss << "detid \t Lorentz angle" << std::endl;
  for( it=detid_la.begin(); it!=detid_la.end(); ++it ) {
    ss << it->first << "\t" << it->second << std::endl;
    ++count;
  }
}

void SiStripLorentzAngle::printSummary(std::stringstream& ss) const
{
  std::map<unsigned int,float> detid_la = getLorentzAngles();
  std::map<unsigned int,float>::const_iterator it;

  SiStripDetSummary summary;

  for( it=detid_la.begin(); it!=detid_la.end(); ++it ) {
    DetId detid(it->first);
    float value = it->second;
    summary.add(detid, value);
  }
  ss << "Summary of lorentz angles:" << std::endl;
  summary.print(ss);

}
