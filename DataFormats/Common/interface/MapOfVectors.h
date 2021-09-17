#ifndef DataFormats_Common_MapOfVectors_h
#define DataFormats_Common_MapOfVectors_h

#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include <vector>
#include <map>

#include <boost/range/iterator_range.hpp>
#include <boost/iterator/iterator_facade.hpp>

class TestMapOfVectors;

namespace edm {

  /* a linearized read-only map-of vectors
   NOTE: The iterator for MapOfVectors an not safely be used across threads, even if only const methods are called.
   */
  template <typename K, typename T>
  class MapOfVectors {
  public:
    typedef MapOfVectors<K, T> self;
    typedef std::map<K, std::vector<T> > TheMap;

    typedef unsigned int size_type;

    typedef std::vector<K> Keys;
    typedef std::vector<size_type> Offsets;
    typedef std::vector<T> Data;

    typedef typename Keys::const_iterator key_iterator;
    typedef Offsets::const_iterator offset_iterator;
    typedef typename Data::const_iterator data_iterator;

    typedef boost::iterator_range<data_iterator> range;

    typedef std::pair<K, range> Pair;

    class Iter : public boost::iterator_facade<Iter, Pair const, boost::forward_traversal_tag> {
    public:
      typedef Iter self;
      Iter() {}

      explicit Iter(key_iterator k, offset_iterator o, std::vector<T> const& d) : key(k), off(o), data(d.begin()) {}

    private:
      friend class boost::iterator_core_access;

      void increment() {
        ++key;
        ++off;
      }

      bool equal(self const& other) const { return this->key == other.key; }

      Pair const& dereference() const {
        // FIXME can be optimized...
        cache.first = *key;
        cache.second = range(data + (*off), data + (*(off + 1)));
        return cache;
      }

      key_iterator key;
      offset_iterator off;
      data_iterator data;
      //This class is not intended to be used across threads
      CMS_SA_ALLOW mutable Pair cache;
    };

    typedef Iter const_iterator;

    range emptyRange() const { return range(m_data.end(), m_data.end()); }

    MapOfVectors() : m_offsets(1, 0) {}

    MapOfVectors(TheMap const& it) {
      m_keys.reserve(it.size());
      m_offsets.reserve(it.size() + 1);
      m_offsets.push_back(0);
      size_type tot = 0;
      for (typename TheMap::const_iterator p = it.begin(); p != it.end(); ++p)
        tot += (*p).second.size();
      m_data.reserve(tot);
      for (typename TheMap::const_iterator p = it.begin(); p != it.end(); ++p)
        loadNext((*p).first, (*p).second);
    }

    void loadNext(K const& k, std::vector<T> const& v) {
      m_keys.push_back(k);
      m_data.resize(m_offsets.back() + v.size());
      std::copy(v.begin(), v.end(), m_data.begin() + m_offsets.back());
      m_offsets.push_back(m_data.size());
    }

    size_type size() const { return m_keys.size(); }

    bool empty() const { return m_keys.empty(); }

    key_iterator findKey(K const& k) const {
      std::pair<key_iterator, key_iterator> p = std::equal_range(m_keys.begin(), m_keys.end(), k);
      return (p.first != p.second) ? p.first : m_keys.end();
    }

    size_type offset(K const& k) const {
      key_iterator p = findKey(k);
      if (p == m_keys.end())
        return m_data.size();
      return m_offsets[p - m_keys.begin()];
    }

    range find(K const& k) const {
      key_iterator p = findKey(k);
      if (p == m_keys.end())
        return emptyRange();
      size_type loc = p - m_keys.begin();
      data_iterator b = m_data.begin() + m_offsets[loc];
      data_iterator e = m_data.begin() + m_offsets[loc + 1];
      return range(b, e);
    }

    ///The iterator returned can not safely be used across threads
    const_iterator begin() const { return const_iterator(m_keys.begin(), m_offsets.begin(), m_data); }

    const_iterator end() const { return const_iterator(m_keys.end(), m_offsets.begin() + m_keys.size(), m_data); }

    void swap(MapOfVectors& other) {
      m_keys.swap(other.m_keys);
      m_offsets.swap(other.m_offsets);
      m_data.swap(other.m_data);
    }

    MapOfVectors& operator=(MapOfVectors const& rhs) {
      MapOfVectors temp(rhs);
      this->swap(temp);
      return *this;
    }

  private:
    //for testing
    friend class ::TestMapOfVectors;

    std::vector<K> m_keys;
    std::vector<size_type> m_offsets;
    std::vector<T> m_data;
  };

  // Free swap function
  template <typename K, typename T>
  inline void swap(MapOfVectors<K, T>& lhs, MapOfVectors<K, T>& rhs) {
    lhs.swap(rhs);
  }

}  // namespace edm

#endif  // DatFormats_Common_MapOfVectors_h
