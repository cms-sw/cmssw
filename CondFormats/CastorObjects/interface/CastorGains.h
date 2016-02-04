#ifndef CastorGains_h
#define CastorGains_h

/** 
\class CastorGains
\author Radek Ofierzynski
Modified by L.Mundim (Mar/2009)
POOL container to store Gain values 4xCapId
*/

#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"
#include "CondFormats/CastorObjects/interface/CastorGain.h"

//typedef CastorCondObjectContainer<CastorGain> CastorGains;

class CastorGains: public CastorCondObjectContainer<CastorGain>
{
 public:
  CastorGains():CastorCondObjectContainer<CastorGain>() {}

  std::string myname() const {return (std::string)"CastorGains";}

 private:
};

#endif
