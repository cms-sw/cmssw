#ifndef CondFormats_EcalObjects_EcalIntercalibConstants_H
#define CondFormats_EcalObjects_EcalIntercalibConstants_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/
#include <map>
#include <boost/cstdint.hpp>


class EcalIntercalibConstants {
  public:
   typedef float EcalIntercalibConstant;
   typedef std::map<uint32_t, EcalIntercalibConstant> EcalIntercalibConstantMap;

    EcalIntercalibConstants();
    ~EcalIntercalibConstants();
    void  setValue(const uint32_t& id, const EcalIntercalibConstant& value);
    const EcalIntercalibConstantMap& getMap() const { return map_; }

  private:
    EcalIntercalibConstantMap map_;
};
#endif
