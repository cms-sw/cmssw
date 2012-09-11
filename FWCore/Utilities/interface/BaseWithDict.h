#ifndef FWCore_Utilities_BaseWithDict_h
#define FWCore_Utilities_BaseWithDict_h

/*----------------------------------------------------------------------
  
BaseWithDict:  A holder for a base class

----------------------------------------------------------------------*/

#include <string>

#include "Reflex/Base.h"

namespace edm {

  class TypeWithDict;

  class BaseWithDict {
  public:
    BaseWithDict() : base_() {}

    explicit BaseWithDict(Reflex::Base const& base) : base_(base) {}

    std::string name() const;

    TypeWithDict toType() const;

    bool isPublic() const;

  private:

    Reflex::Base base_;
  };

}
#endif
