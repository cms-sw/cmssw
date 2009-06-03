#ifndef Cond_IOVKeysDescription_h
#define Cond_IOVKeysDescription_h

#include "CondFormats/Common/interface/SmallWORMDict.h"

namespace cond {

  /*
   * Describe the fixed set of keys to be used in a keylist 
   */
  class IOVKeysDescription : public IOVDescription {
  public:
    IOVKeysDescription(){}
    explicit IOVKeysDescription(std::vector<std::string> const & idict) :
      dict_m(idict){}

    virtual ~IOVKeysDescription(){}
    virtual IOVKeysDescription * clone() const { return new  IOVKeysDescription(*this);}

    SmallWORMDict const & dict { return dict_m;}

  private:

    SmallWORMDict dict_m;

  };


}

#endif
