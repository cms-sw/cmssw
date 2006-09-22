#ifndef InitRootHandlers_H
#define InitRootHandlers_H

#include <string>

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
}

namespace edm {
namespace service {
class InitRootHandlers
{
private:
  bool unloadSigHandler;
  bool resetErrHandler;

public:
  InitRootHandlers (edm::ParameterSet const& pset, edm::ActivityRegistry & activity);
  ~InitRootHandlers ();

};
}  // end of namespace service
}  // end of namespace edm

#endif // InitRootHandlers_H
