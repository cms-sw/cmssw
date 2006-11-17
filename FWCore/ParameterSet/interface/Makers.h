#ifndef ParameterSet_Makers_h
#define ParameterSet_Makers_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/shared_ptr.hpp"

#include <string>

namespace edm {
   namespace pset {
      boost::shared_ptr<edm::ParameterSet> 
      makePSet(const std::string & s);

      boost::shared_ptr<edm::ParameterSet>
      makeDefaultPSet(const edm::FileInPath & fip);

   }
}
#endif
