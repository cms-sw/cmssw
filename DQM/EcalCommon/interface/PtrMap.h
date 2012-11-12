#ifndef PtrMap_h
#define PtrMap_h

#include <map>

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  template <class Key, class V>
  class PtrMap : public std::map<Key, V*> {
    typedef V* T;
    typedef std::map<Key, T> BaseType;
  public:
    PtrMap() : std::map<Key, T>() {}
    ~PtrMap()
    {
      for(typename BaseType::iterator itr(this->begin()); itr != this->end(); ++itr)
        delete itr->second;
    }
    T& operator[](Key const& _k)
    {
      typename BaseType::iterator itr(this->find(_k));
      if(itr == this->end())
        throw cms::Exception("InvalidReference") << "Pointer for key " << _k << " not found";

      return itr->second;
    }
    void clear()
    {
      for(typename BaseType::iterator itr(this->begin()); itr != this->end(); ++itr)
        delete itr->second;
      BaseType::clear();
    }
    void erase(typename BaseType::iterator _itr)
    {
      delete _itr->second;
      BaseType::erase(_itr);
    }
    size_t erase(Key const& _k)
    {
      typename BaseType::iterator itr(this->find(_k));
      if(itr == this->end()) return 0;
      delete itr->second;
      BaseType::erase(itr);
      return 1;
    }
    void erase(typename BaseType::iterator _first, typename BaseType::iterator _last)
    {
      for(typename BaseType::iterator itr(_first); itr != _last; ++itr)
        delete itr->second;
      BaseType::erase(_first, _last);
    }

    std::pair<typename BaseType::iterator, bool> insert(Key const& _k, T _v)
    {
      return BaseType::insert(std::pair<Key, T>(_k, _v));
    }
  };

}

#endif
