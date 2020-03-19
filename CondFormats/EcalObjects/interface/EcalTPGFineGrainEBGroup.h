#ifndef EcalTPGFineGrainEBGroup_h
#define EcalTPGFineGrainEBGroup_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"

/*

P.P.
*/

class EcalTPGFineGrainEBGroup : public EcalTPGGroups {
public:
  EcalTPGFineGrainEBGroup();
  ~EcalTPGFineGrainEBGroup();

  COND_SERIALIZABLE;
};

#endif
