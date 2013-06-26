#ifndef FWCore_Utilities_Map_h
#define FWCore_Utilities_Map_h

#include <cassert>
#include <map>

// Alternatives to std::map::operator[]
// They differ on what happens if the element is not in the map.
// Those that do not insert into the map will also work on a const map.

namespace edm {

  // This silly little function documents the fact
  // a possible insert into the map is intentional.
  template <typename Key, typename Value>
  inline
  Value&
  findOrInsert(std::map<Key, Value>& m, Key const& k) {
    return m[k];
  }

  // This function will not insert into the map.
  // If the element is not found, it returns the supplied default value
  // Comes in const and non-const versions
  template <typename Key, typename Value>
  inline
  Value const&
  findOrDefault(std::map<Key, Value> const& m, Key const& k, Value const& defaultValue) {
    typename std::map<Key, Value>::const_iterator it = m.find(k);
    return (it == m.end() ? defaultValue : it->second);
  }

  template <typename Key, typename Value>
  inline
  Value&
  findOrDefault(std::map<Key, Value>& m, Key const& k, Value& defaultValue) {
    typename std::map<Key, Value>::const_iterator it = m.find(k);
    return (it == m.end() ? defaultValue : it->second);
  }

  // This function will not insert into the map.
  // If the element is not found, it returns a default constructed value
  // Note that the return is by value, so if the element is found, it is copied.
  template <typename Key, typename Value>
  inline
  Value
  findOrDefault(std::map<Key, Value> const& m, Key const& k) {
    typename std::map<Key, Value>::const_iterator it = m.find(k);
    return (it == m.end() ? Value() : it->second);
  }

  // This function will not insert into the map.
  // If the element is not found, it asserts.
  // Comes in const and non-const versions
  template <typename Key, typename Value>
  inline
  Value const&
  findOrAssert(std::map<Key, Value> const& m, Key const& k) {
    typename std::map<Key, Value>::const_iterator it = m.find(k);
    if (it == m.end()) assert("findOrAssert" && 0);
    return it->second;
  }

  template <typename Key, typename Value>
  inline
  Value&
  findOrAssert(std::map<Key, Value>& m, Key const& k) {
    typename std::map<Key, Value>::const_iterator it = m.find(k);
    if (it == m.end()) assert("findOrAssert" && 0);
    return it->second;
  }
}
#endif
