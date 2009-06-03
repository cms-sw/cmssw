#ifndef CondCore_IOVService_KeyList_h
#define CondCore_IOVService_KeyList_h

#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "DataSvc/Ref.h"
#include<vector>
#include<string>

/*
 * KeyList represents a set of payloads each identified by a key  and "valid" at given time
 * Usually these payloads are configuration objects loaded in anvance
 * The model used here calls for all payloads to be "stored" in a single IOVSequence each identified by a unique key 
 * (properly hashed to be mapped in 64bits)
 *
 * the keylist is just a vector of the hashes each corresponding to a key
 * the correspondence position in the vector user-friendly name is kept in 
 * a list of all "names" that is defined in advance and kept in a dictionary at IOVSequence level
 
 *
 */

namespace cond {
   
   class IOVKeysDescription;

  class KeyList {
  public:
    typedef cond::DataWrapper<BaseKeyed> Wrapper;

    KeyList(IOVKeysDescription const * idescr);

    void load(std::vector<unsigned long long> const & keys);

    template<typename T> 
    T const * get(char const * iname) const {
      return dynamic_cast<T const *>(elem(iname));
    }
    
    template<typename T> 
    T const * get(std::string const & iname) const {
      return dynamic_cast<T const *>(elem(iname));
    }
    
    BaseKeyed const * elem(char const * iname) const;

    BaseKeyed const * elem(std::string const & iname) const;

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
