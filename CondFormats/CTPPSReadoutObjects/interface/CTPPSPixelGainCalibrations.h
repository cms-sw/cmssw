#ifndef CondFormats_CTPPSReadoutObjects_CTPPSPixelGainCalibrations_h
#define CondFormats_CTPPSReadoutObjects_CTPPSPixelGainCalibrations_h

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibration.h"
#include <map>
#include <vector>


class CTPPSPixelGainCalibrations{
 public:
  typedef std::map<uint32_t,CTPPSPixelGainCalibration> CalibMap;

  CTPPSPixelGainCalibrations(){}
  virtual ~CTPPSPixelGainCalibrations(){}

  void setGainCalibration(const uint32_t& DetId, const CTPPSPixelGainCalibration & PixGains);
  void setGainCalibration(const uint32_t& DetId, const std::vector<float>& peds, const std::vector<float>& gains);
  void setGainCalibrations(const CalibMap & PixGainsCalibs);
  void setGainCalibrations(const std::vector<uint32_t>& detidlist, const std::vector<std::vector<float>>& peds, const std::vector<std::vector<float>>& gains);

  const CalibMap & getCalibMap()const { return m_calibrations;}

  CTPPSPixelGainCalibration getGainCalibration(const uint32_t & detid) const;
  CTPPSPixelGainCalibration & getGainCalibration(const uint32_t & detid);

  
  
  int size() const {return m_calibrations.size();}

 private:
  CalibMap m_calibrations;

  COND_SERIALIZABLE;
};

#endif
