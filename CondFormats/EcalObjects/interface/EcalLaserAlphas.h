#ifndef CondFormats_EcalObjects_EcalLaserAlphas_H
#define CondFormats_EcalObjects_EcalLaserAlphas_H
/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 15 May 2007
 * $Id: EcalLaserAlphas.h,v 1.2 2007/07/16 22:01:29 meridian Exp $
 **/
#include <vector>
#include <boost/cstdint.hpp>

class EcalLaserAlphas {
  public:
   typedef float EcalLaserAlpha;
   typedef std::vector<EcalLaserAlpha> EcalLaserAlphaMap;

    EcalLaserAlphas();
    ~EcalLaserAlphas();

    void  setValue(int hashedIndex, const EcalLaserAlpha& value) { map_[hashedIndex] = value; };
    const EcalLaserAlphaMap& getMap() const { return map_; }

  private:
    EcalLaserAlphaMap map_;
};

#endif
