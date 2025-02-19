#ifndef CastorChannelQuality_h
#define CastorChannelQuality_h

/** 
\class CastorChannelQuality
\author Radek Ofierzynski
POOL object to store CastorChannelStatus
*/

#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"
#include "CondFormats/CastorObjects/interface/CastorChannelStatus.h"

//typedef CastorCondObjectContainer<CastorChannelStatus> CastorChannelQuality;

class CastorChannelQuality: public CastorCondObjectContainer<CastorChannelStatus>
{
 public:
  CastorChannelQuality():CastorCondObjectContainer<CastorChannelStatus>() {}

  std::string myname() const {return (std::string)"CastorChannelQuality";}

 private:
};


#endif

