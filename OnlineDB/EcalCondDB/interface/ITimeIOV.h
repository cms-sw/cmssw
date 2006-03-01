#ifndef ITIMEIOV_H
#define ITIMEIOV_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/ITag.h"
#include "OnlineDB/EcalCondDB/interface/IIOV.h"

class ITimeIOV {
 public:
  virtual void fetchAt(IIOV* fillIOV, const Tm eventTm, ITag* tag) const throw(std::runtime_error) =0;

  virtual void fetchWithin(std::vector<IIOV*>* fillVec, const Tm beginTm, const Tm endTm, ITag* tag) const throw(std::runtime_error) =0;
  
};

#endif
