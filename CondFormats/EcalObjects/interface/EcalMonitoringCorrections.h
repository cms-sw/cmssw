#ifndef CondFormats_EcalObjects_EcalMonitoringCorrections_H
#define CondFormats_EcalObjects_EcalMonitoringCorrections_H
/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: $
 **/
#include <map>
#include <boost/cstdint.hpp>


class EcalMonitoringCorrections {
  public:
   typedef float EcalMonitoringCorrection;
   typedef std::map<uint32_t, EcalMonitoringCorrection> EcalMonitoringCorrectionMap;

    EcalMonitoringCorrections();
    ~EcalMonitoringCorrections();
    void  setValue(const uint32_t& id, const EcalMonitoringCorrection& value);
    const EcalMonitoringCorrectionMap& getMap() const { return map_; }
    
  private:
    EcalMonitoringCorrectionMap map_;
};
#endif
