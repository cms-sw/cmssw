#ifndef CondFormats_EcalObjects_EcalLaserAPDPNRatiosRef_H
#define CondFormats_EcalObjects_EcalLaserAPDPNRatiosRef_H
/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 15 May 2007
 * $Id: EcalLaserAPDPNRatiosRef.cc,v 1.1 2007/05/16 11:46:00 vladlen Exp $
 **/
#include <map>
#include <boost/cstdint.hpp>


class EcalLaserAPDPNRatiosRef {
  public:
   typedef float EcalLaserAPDPNref;
   typedef std::map<uint32_t, EcalLaserAPDPNref> EcalLaserAPDPNRatiosRefMap;

   EcalLaserAPDPNRatiosRef();
   ~EcalLaserAPDPNRatiosRef();
   void  setValue(const uint32_t& id, const EcalLaserAPDPNref& value);
   const EcalLaserAPDPNRatiosRefMap& getMap() const { return map_; }

  private:
    EcalLaserAPDPNRatiosRefMap map_;
};
#endif
