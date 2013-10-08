#ifndef DataFormats_Provenance_ParentageRegistry_h
#define DataFormats_Provenance_ParentageRegistry_h

#include "tbb/concurrent_unordered_map.h"

#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageID.h"


// Note that this registry is *not* directly persistable. The contents
// are persisted, but not the container.
namespace edm
{
  class ParentageRegistry {
  public:
    typedef edm::ParentageID   key_type;
    typedef edm::Parentage     value_type;
    
    static ParentageRegistry* instance();
    
    /// Retrieve the value_type object with the given key.
    /// If we return 'true', then 'result' carries the
    /// value_type object.
    /// If we return 'false, no matching key was found, and
    /// the value of 'result' is undefined.
    bool getMapped(key_type const& k, value_type& result) const;
    
    /** Retrieve a pointer to the value_type object with the given key.
     If there is no object associated with the given key 0 is returned.
     */
    value_type const* getMapped(key_type const& k) const;
    
    /// Insert the given value_type object into the
    /// registry. If there was already a value_type object
    /// with the same key, we don't change it. This should be OK,
    /// since it should have the same contents if the key is the
    /// same.  Return 'true' if we really added the new
    /// value_type object, and 'false' if the
    /// value_type object was already present.
    bool insertMapped(value_type const& v);
    
    ///Not thread safe
    void clear();

    struct key_hash {
      std::size_t operator()(key_type const& iKey) const{
        return iKey.smallHash();
      }
    };
  private:
    tbb::concurrent_unordered_map<key_type,value_type,key_hash> m_map;
  };

}

#endif
