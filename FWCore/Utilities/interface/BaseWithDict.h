#ifndef FWCore_Utilities_BaseWithDict_h
#define FWCore_Utilities_BaseWithDict_h

/*----------------------------------------------------------------------
  
BaseWithDict:  A holder for a base class

----------------------------------------------------------------------*/

#include <string>

class TBaseClass;

namespace edm {

  class TypeWithDict;

  class BaseWithDict {
  public:
    BaseWithDict();

    explicit BaseWithDict(TBaseClass* baseClass);

    std::string name() const;

    TypeWithDict toType() const;

    bool isPublic() const;

  private:

    TBaseClass* baseClass_;
  };

}
#endif
