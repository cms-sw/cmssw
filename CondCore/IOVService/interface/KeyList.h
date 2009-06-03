#ifndef CondCore_IOVService_KeyList_h
#define CondCore_IOVService_KeyList_h

#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondFormats/Common/interface/BaseKeyed.h"


/*
 * KeyList represents a set of payloads identified by a keys and "valid" at given time
 * Usually these payloads are configuration objects loaded in anvance
 * The model used here calls for all payloads to be "stored" in a single IOVSequence each identified by a unique name (properly hashed to be mapped in 64bits
 *
 * the list of all keys are defined in advance and kept in a dictionary at IOVSequence level
 * the keylist is just a vector of the hashes each corresponding to a key 
 *
 */

namespace cond {

  class KeyList {
  public:
    typedef cond::DataWrapper<BaseKeyed> Wrapper;

    KeyList(IOVKeysDescription const * idescr);

    void load(std::vector<unsigned long long> const & names);

    template<typename T> 
    T const * get(char const * ikey) const {
      return dynamic_cast<T const *>(elem(ikey));
    }
    
    template<typename T> 
    T const * get(std::string const & ikey) const {
      return dynamic_cast<T const *>(elem(ikey));
    }
    
    BaseKeyed const * elem(char const * ikey) const;

    BaseKeyed const * elem(std::string const & ikey) const;

  private:
    // tag and dict
    IOVKeysDescription const * m_description;

    // the full collection of keyed object
    cond::IOVProxy m_sequence;
    // the current set
    std::vector<pool::Ref<Wrapper> > m_data;

  };


}

#endif
