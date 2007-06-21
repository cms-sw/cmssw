#ifndef CondFormats_EcalObjects_EcalLaserAlphas_H
#define CondFormats_EcalObjects_EcalLaserAlphas_H
/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 15 May 2007
 * $Id: EcalLaserAlphas.h,v 1.0 2007/05/15 12:28:34 vladlen Exp $
 **/
#include <map>
#include <boost/cstdint.hpp>


class EcalLaserAlphas {
  public:
   typedef float EcalLaserAlpha;
   typedef std::map<uint32_t, EcalLaserAlpha> EcalLaserAlphaMap;

    EcalLaserAlphas();
    ~EcalLaserAlphas();
    void  setValue(const uint32_t& id, const EcalLaserAlpha& value);
    const EcalLaserAlphaMap& getMap() const { return map_; }

  private:
    EcalLaserAlphaMap map_;
};
#endif
