#ifndef CondFormats_EcalObjects_EcalLaserAPDPNRatiosRef_H
#define CondFormats_EcalObjects_EcalLaserAPDPNRatiosRef_H
/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 10 July 2007
 * $Id: EcalLaserAPDPNRatiosRef.h,v 1.2 2007/07/16 22:01:29 meridian Exp $
 **/
#include <vector>
#include <boost/cstdint.hpp>


class EcalLaserAPDPNRatiosRef {
  public:
   typedef float EcalLaserAPDPNref;
   typedef std::vector<EcalLaserAPDPNref> EcalLaserAPDPNRatiosRefMap;

   EcalLaserAPDPNRatiosRef();
   ~EcalLaserAPDPNRatiosRef();

   void  setValue(int hashedIndex, const EcalLaserAPDPNref& value) { map_[hashedIndex] = value; };
   const EcalLaserAPDPNRatiosRefMap& getMap() const { return map_; }

  private:
    EcalLaserAPDPNRatiosRefMap map_;
};


#endif
