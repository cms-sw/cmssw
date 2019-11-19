
#ifndef CondCore_CondDB_KeyList_h
#define CondCore_CondDB_KeyList_h

#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/Binary.h"
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
//
#include <map>
#include <memory>
#include <vector>
#include <string>

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

  namespace persistency {

    class KeyList {
    public:
      ///Called by PoolDBESSource
      void init(IOVProxy iovProxy);
      void init(KeyList const&);

      /// determines which keys to use to read from the DB. Should only be used by PoolDBESSource
      void setKeys(const std::vector<unsigned long long>& keys);

      ///Retrieves the pre-fetched data. The index is the same order as the keys used in setKeys.
      template <typename T>
      std::shared_ptr<T> getUsingIndex(size_t n) const {
        if (n >= size())
          throwException("Index outside the bounds of the key array.", "KeyList::getUsingIndex");
        if (m_keys[n] == 0 or m_data[n].first.empty()) {
          throwException("Payload for index " + std::to_string(n) + " has not been found.", "KeyList::getUsingIndex");
        }
        auto const& i = m_data[n];
        return deserialize<T>(i.first, i.second.first, i.second.second);
      }

      /** Retrieve the item associated with the key directly from the DB.
          The function is non-const since it is not thread-safe. */
      template <typename T>
      std::shared_ptr<T> getUsingKey(unsigned long long key) const {
        auto item = loadFromDB(key);
        return deserialize<T>(item.first, item.second.first, item.second.second);
      }

      ///Number of keys based on container passed to setKeys.
      size_t size() const { return m_data.size(); }

    private:
      std::pair<std::string, std::pair<cond::Binary, cond::Binary>> loadFromDB(unsigned long long key) const;
      // the db session, protected by a mutex
      mutable IOVProxy m_proxy;
      // the key selection:
      std::vector<unsigned long long> m_keys;
      std::vector<std::pair<std::string, std::pair<cond::Binary, cond::Binary>>> m_data;
    };

  }  // namespace persistency
}  // namespace cond

#endif
