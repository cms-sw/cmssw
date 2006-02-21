#ifndef CondFormats_EcalObjects_EcalIntercalibConstants_H
#define CondFormats_EcalObjects_EcalIntercalibConstants_H

#include <map>
#include <boost/cstdint.hpp>


typedef float EcalIntercalibConstant;
typedef std::map<uint32_t, EcalIntercalibConstant> EcalIntercalibConstantMap;

class EcalIntercalibConstants {
  public:
    EcalIntercalibConstants();
    ~EcalIntercalibConstants();
    void  setValue(const uint32_t& id, const EcalIntercalibConstant& value);
    const EcalIntercalibConstantMap& getMap() const { return map_; }

  private:
    EcalIntercalibConstantMap map_;
};
#endif
