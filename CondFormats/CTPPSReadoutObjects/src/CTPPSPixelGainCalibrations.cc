#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

void CTPPSPixelGainCalibrations::setGainCalibration(const uint32_t& detid, const CTPPSPixelGainCalibration & PixGains){
  m_calibrations[detid] = PixGains;
}

void CTPPSPixelGainCalibrations::setGainCalibration(const uint32_t& detid, const vector<float> & peds, const vector<float> & gains){
  m_calibrations[detid] =  CTPPSPixelGainCalibration(detid,peds,gains);
}

void CTPPSPixelGainCalibrations::setGainCalibrations(const calibmap & PixGainsCalibs){
  m_calibrations = PixGainsCalibs;
}

void CTPPSPixelGainCalibrations::setGainCalibrations(const vector<uint32_t>& detidlist, const vector<vector<float>>& peds, const vector<vector<float>>& gains){
  int nids=detidlist.size();
  for (int detid=0; detid<nids ; ++detid){
    const  vector<float>& pedsvec  = peds[detid];
    const  vector<float>& gainsvec = gains[detid];
    m_calibrations[detid]=  CTPPSPixelGainCalibration(detid,pedsvec,gainsvec);
  }
}


CTPPSPixelGainCalibration & CTPPSPixelGainCalibrations::getGainCalibration(const uint32_t & detid) { // returns the ref and if does not exist it creates
  return m_calibrations[detid];
}

CTPPSPixelGainCalibration  CTPPSPixelGainCalibrations::getGainCalibration(const uint32_t & detid) const{ // returns the object does not change the map
  calibmap::const_iterator it = m_calibrations.find(detid);
  if (it != m_calibrations.end())
    return it->second;

  else
    edm::LogError("CTPPSPixelGainCalibrations")<< "No gain calibrations defined for detid " << detid << "\n";
// throw cms::Exception("CTPPSPixelGainCalibrations")<< "No gain calibrations defined for detid " << detid << "\n";

  CTPPSPixelGainCalibration a;

  return a;

}



