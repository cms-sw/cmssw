#ifndef CastorRecoParams_h
#define CastorRecoParams_h


#include "CondFormats/CastorObjects/interface/CastorRecoParam.h"
#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"


class CastorRecoParams: public CastorCondObjectContainer<CastorRecoParam>
{
 public:
  CastorRecoParams():CastorCondObjectContainer<CastorRecoParam>() {}

  std::string myname() const {return (std::string)"CastorRecoParams";}

 private:

};
#endif
