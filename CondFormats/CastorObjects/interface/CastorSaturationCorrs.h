#ifndef CastorSaturationCorrs_h
#define CastorSaturationCorrs_h


#include "CondFormats/CastorObjects/interface/CastorSaturationCorr.h"
#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"


class CastorSaturationCorrs: public CastorCondObjectContainer<CastorSaturationCorr>
{
 public:
  CastorSaturationCorrs():CastorCondObjectContainer<CastorSaturationCorr>() {}

  std::string myname() const {return (std::string)"CastorSaturationCorrs";}

 private:

};
#endif
