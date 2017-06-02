#ifndef CondFormats_CTPPSReadoutObjects_CTPPSPixelGainCalibrations_h
#define CondFormats_CTPPSReadoutObjects_CTPPSPixelGainCalibrations_h

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibration.h"
#include <map>
#include <vector>

using namespace std;

class CTPPSPixelGainCalibrations{
 public:
  typedef map<uint32_t,CTPPSPixelGainCalibration> calibmap;

  CTPPSPixelGainCalibrations(){}
  virtual ~CTPPSPixelGainCalibrations(){}

  void setGainCalibration(const uint32_t& DetId, const CTPPSPixelGainCalibration & PixGains);
  void setGainCalibration(const uint32_t& DetId, const vector<float>& peds, const vector<float>& gains);
  void setGainCalibrations(const calibmap & PixGainsCalibs);
  void setGainCalibrations(const vector<uint32_t>& detidlist, const vector<vector<float>>& peds, const vector<vector<float>>& gains);

  const calibmap & getCalibmap()const { return m_calibrations;}

  CTPPSPixelGainCalibration getGainCalibration(const uint32_t & detid) const;
  CTPPSPixelGainCalibration & getGainCalibration(const uint32_t & detid);

  
  
  int size() const {return m_calibrations.size();}

 private:
  calibmap m_calibrations;

  COND_SERIALIZABLE;
};

#endif
