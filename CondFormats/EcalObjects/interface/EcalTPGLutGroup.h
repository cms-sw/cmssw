#ifndef EcalTPGLutGroup_h
#define EcalTPGLutGroup_h

#include "CondFormats/Common/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"

/*

P.P.
*/



class EcalTPGLutGroup : public EcalTPGGroups
{
 public:

  EcalTPGLutGroup() ;
  ~EcalTPGLutGroup() ;


 COND_SERIALIZABLE;
};

#endif
