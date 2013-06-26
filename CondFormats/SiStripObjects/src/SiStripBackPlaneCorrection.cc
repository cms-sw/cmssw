#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 

bool SiStripBackPlaneCorrection::putBackPlaneCorrection(const uint32_t& detid, float value){
  std::map<unsigned int,float>::const_iterator id=m_BPC.find(detid);
  if(id!=m_BPC.end()){
    edm::LogError("SiStripBackPlaneCorrection") << "SiStripBackPlaneCorrection for DetID " << detid << " is already stored. Skippig this put" << std::endl;
    return false;
  }
  else m_BPC[detid]=value;
  return true;
}
float SiStripBackPlaneCorrection::getBackPlaneCorrection(const uint32_t& detid) const  {
  std::map<unsigned int,float>::const_iterator id=m_BPC.find(detid);
  if(id!=m_BPC.end()) return id->second;
  else {
    edm::LogError("SiStripBackPlaneCorrection") << "SiStripBackPlaneCorrection for DetID " << detid << " is not stored" << std::endl; 
  }
  return 0;
}

void SiStripBackPlaneCorrection::printDebug(std::stringstream& ss) const
{
  std::map<unsigned int,float> detid_la = getBackPlaneCorrections();
  std::map<unsigned int,float>::const_iterator it;
  size_t count=0;
  ss << "SiStripBackPlaneCorrectionReader:" << std::endl;
  ss << "detid \t Geometry \t Back Plane Corrections" << std::endl;
  for( it=detid_la.begin(); it!=detid_la.end(); ++it ) {
    SiStripDetId SSdetId(it->first);
    unsigned int moduleGeometry = SSdetId.moduleGeometry();
    ss << it->first << "\t" << moduleGeometry << "\t" << it->second << std::endl;
    ++count;
  }
}

void SiStripBackPlaneCorrection::printSummary(std::stringstream& ss) const
{
  std::map<unsigned int,float> detid_la = getBackPlaneCorrections();
  std::map<unsigned int,float>::const_iterator it;

  SiStripDetSummary summary;

  for( it=detid_la.begin(); it!=detid_la.end(); ++it ) {
    DetId detid(it->first);
    float value = it->second;
    summary.add(detid, value);
  }
  ss << "Summary of BackPlane corrections:" << std::endl;
  summary.print(ss);

}
