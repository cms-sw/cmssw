#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"

EcalTPGGroups::EcalTPGGroups()
{ }

EcalTPGGroups::~EcalTPGGroups()
{ }

void  EcalTPGGroups::setValue(const uint32_t & rawId, const   uint32_t & ObjectId)
{ 
  map_[rawId] = ObjectId ;
}

