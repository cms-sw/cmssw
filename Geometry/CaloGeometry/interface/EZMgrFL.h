#ifndef GEOMETRY_CALOGEOMETRY_EZMGRFL_H
#define GEOMETRY_CALOGEOMETRY_EZMGRFL_H 1

#include <vector>
#include <cassert>

template <class T>
class EZMgrFL {
public:
  typedef std::vector<T> VecType;
  typedef typename VecType::iterator iterator;
  typedef typename VecType::const_iterator const_iterator;
  typedef typename VecType::reference reference;
  typedef typename VecType::const_reference const_reference;
  typedef typename VecType::value_type value_type;
  typedef typename VecType::size_type size_type;

  EZMgrFL(size_type vecSize, size_type subSize) : m_vecSize(vecSize), m_subSize(subSize) {
    m_vec.resize(0);
    assert(subSize > 0);
    assert(vecSize > 0);
    assert(0 == m_vec.capacity());
  }

  iterator reserve() { return assign(); }
  iterator resize() { return assign(); }

  iterator assign(const T& t = T()) {
    assert((m_vec.size() + m_subSize) <= m_vecSize);
    if (0 == m_vec.capacity()) {
      m_vec.reserve(m_vecSize);
      assert(m_vecSize == m_vec.capacity());
    }
    for (size_type i(0); i != m_subSize; ++i) {
      m_vec.emplace_back(t);
    }
    return (m_vec.end() - m_subSize);
  }

  size_type subSize() const { return m_subSize; }
  size_type size() const { return m_vec.size(); }

  size_type vecSize() const { return m_vec.size(); }

  EZMgrFL() = delete;
  EZMgrFL(const EZMgrFL&) = delete;
  EZMgrFL& operator=(const EZMgrFL&) = delete;

private:
  const size_type m_vecSize;
  const size_type m_subSize;
  VecType m_vec;
};

#endif
