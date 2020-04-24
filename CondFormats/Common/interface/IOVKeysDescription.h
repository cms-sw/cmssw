#ifndef Cond_IOVKeysDescription_h
#define Cond_IOVKeysDescription_h

#include "CondFormats/Common/interface/IOVDescription.h"
#include "CondFormats/Common/interface/SmallWORMDict.h"
#include <string>

namespace cond {

  /*
   * Describe the fixed set of keys to be used in a keylist 
   */
  class IOVKeysDescription : public IOVDescription {
  public:
    IOVKeysDescription(){}
    explicit IOVKeysDescription(std::vector<std::string> const & idict, std::string const & itag) :
      dict_m(idict), m_tag(itag){}

    virtual ~IOVKeysDescription(){}
    virtual IOVKeysDescription * clone() const { return new  IOVKeysDescription(*this);}

    // the associated "tag"
    std::string const & tag() const { return m_tag; }
 

    // the list of keys
    SmallWORMDict const & dict() const { return dict_m;}

  private:
    SmallWORMDict dict_m;
    std::string m_tag;

  
 COND_SERIALIZABLE;
};


}

#endif
