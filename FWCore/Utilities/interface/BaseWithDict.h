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
  // NOTE: Any use of class BaseWithDict in ROOT 5 is not thread safe without use
  // of a properly scoped CINT mutex as in this example:
  // {
  //   R__LOCKGUARD(gCintMutex);
  //   TypeBases bases(myType);
  //   for (auto const& b : bases) {
  //     BaseWithDict base(b);
  //     ...
  //   }
  //   //  other use of bases goes here
  // }
  // The situation in ROOT 6 is not yet determined.
  public:
    BaseWithDict();

    explicit BaseWithDict(TBaseClass* baseClass);

    std::string name() const;

    TypeWithDict typeOf() const;

    bool isPublic() const;

  private:

    TBaseClass* baseClass_;
  };

}
#endif
