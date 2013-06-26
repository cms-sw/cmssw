#ifndef IRUNIOV_H
#define IRUNIOV_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/ITag.h"

typedef int run_t;

class IRunIOV {
 public:
  virtual void fetchAt(IIOV* fillIOV, const run_t run, ITag* tag) const throw(std::runtime_error) =0;

  virtual void fetchWithin(std::vector<IIOV>* fillVec, const run_t beginRun, const run_t endRun, ITag* tag) const throw(std::runtime_error) =0;
  
};

#endif
