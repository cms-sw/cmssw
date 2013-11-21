#ifndef FWCore_ParameterSet_Registry_h
#define FWCore_ParameterSet_Registry_h

// ----------------------------------------------------------------------
// Declaration for pset::Registry. This is an implementation detail of
// the ParameterSet library.
//
// A Registry is used to keep track of the persistent form of all
// ParameterSets used a given program, so that they may be retrieved by
// ParameterSetID, and so they may be written to persistent storage.
// Note that this registry is *not* directly persistable. The contents
// are persisted, but not the container.
// ----------------------------------------------------------------------

#include <iosfwd>
#include <map>
#include "tbb/concurrent_unordered_map.h"

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  namespace pset {

    class Registry {
    public:
      typedef edm::ParameterSetID key_type;
      typedef edm::ParameterSet   value_type;

      static Registry* instance();

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
      typedef tbb::concurrent_unordered_map<key_type,value_type,key_hash> map_type;
      typedef map_type::const_iterator const_iterator;

      const_iterator begin() const {
        return m_map.begin();
      }

      const_iterator end() const {
        return m_map.end();
      }

      bool empty() const {
        return m_map.empty();
      }

      size_t size() const {
        return m_map.size();
      }

      /// Fill the given map with the persistent form of each
      /// ParameterSet in the registry.
      typedef std::map<ParameterSetID, ParameterSetBlob> regmap_type;
      void fillMap(regmap_type& fillme) const;

      void print(std::ostream& os) const;

    private:
      map_type m_map;
    };

    /// Associated free functions.

    /// Save the ParameterSetID of the top-level ParameterSet.
    void setID(ParameterSetID const& id);

    /// Return the ParameterSetID of the top-level ParameterSet.
    /// Note the the returned ParameterSetID may be invalid;
    /// this will happen if the Registry has not yet been filled.
    ParameterSetID const& getProcessParameterSetID();
  }  // namespace pset

  ParameterSet const& getProcessParameterSet();

}  // namespace edm

#endif
