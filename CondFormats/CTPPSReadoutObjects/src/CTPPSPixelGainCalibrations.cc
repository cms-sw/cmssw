#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void CTPPSPixelGainCalibrations::setGainCalibration(const uint32_t& detid, const CTPPSPixelGainCalibration & PixGains){

  if(detid!=PixGains.getDetId()){ // if no detid set in the pixgains
    CTPPSPixelGainCalibration newPixGains =    PixGains; //maybe the copy works?
    newPixGains.setIndexes(detid); //private accessor, only friend class can use it
    edm::LogInfo("CTPPSPixelGainCalibrations") << "newPixGains detId = "<< newPixGains.getDetId() 
	      << " ; iBegin = "<< newPixGains.getIBegin()
	      << " ; iEnd = "<< newPixGains.getIEnd()
	      << " ; nCols = "<< newPixGains.getNCols() 
	      << " ; nRows ="<<newPixGains.getNRows();

    int npix = newPixGains.getIEnd() ; 
    //bool dead,noisy;
    if(npix!=0 )
      edm::LogInfo("CTPPSPixelGainCalibrations") 
	<< "newPixGains Ped[0] = "<< newPixGains.getPed(0)
	<< " ; Gain[0] = " << newPixGains.getGain(0)
	<< " ; dead = " << newPixGains.isDead(0) 
	<< " ; noisy = "<< newPixGains.isNoisy(0) ;
    else 
      edm::LogError("CTPPSPixelGainCalibrations") << "looks like setting gain calibrations did not work, npix is "<< npix ;

    m_calibrations[detid]=newPixGains;
  }

  else
    m_calibrations[detid] = PixGains;
}

void CTPPSPixelGainCalibrations::setGainCalibration(const uint32_t& detid, const std::vector<float> & peds, const std::vector<float> & gains){
  m_calibrations[detid] =  CTPPSPixelGainCalibration(detid,peds,gains);
}

void CTPPSPixelGainCalibrations::setGainCalibrations(const CalibMap & PixGainsCalibs){
  m_calibrations = PixGainsCalibs;
}

void CTPPSPixelGainCalibrations::setGainCalibrations(const std::vector<uint32_t>& detidlist, const std::vector<std::vector<float>>& peds, const std::vector<std::vector<float>>& gains){
  int nids=detidlist.size();
  for (int detid=0; detid<nids ; ++detid){
    const  std::vector<float>& pedsvec  = peds[detid];
    const  std::vector<float>& gainsvec = gains[detid];
    m_calibrations[detid]=  CTPPSPixelGainCalibration(detid,pedsvec,gainsvec);
  }
}


CTPPSPixelGainCalibration & CTPPSPixelGainCalibrations::getGainCalibration(const uint32_t & detid) { // returns the ref and if does not exist it creates
  return m_calibrations[detid];
}

CTPPSPixelGainCalibration  CTPPSPixelGainCalibrations::getGainCalibration(const uint32_t & detid) const{ // returns the object does not change the map
  CalibMap::const_iterator it = m_calibrations.find(detid);
  if (it != m_calibrations.end())
    return it->second;

  else
    edm::LogError("CTPPSPixelGainCalibrations")<< "No gain calibrations defined for detid " << detid << "\n";

  CTPPSPixelGainCalibration a;
  return a;

}



