#ifndef DataFormats_Common_DetSetNew_h
#define DataFormats_Common_DetSetNew_h

#include "DataFormats/Common/interface/Ref.h"
#include <vector>
#include <cassert>

namespace edmNew {
  //  FIXME move it elsewhere....
  typedef unsigned int det_id_type;

  template <typename T>
  class DetSetVector;

  /* a proxy to a variable size array of T belonging to
   * a "channel" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   * 
   */
  template <typename T>
  class DetSet {
  public:
    typedef DetSetVector<T> Container;
    typedef unsigned int size_type;  // for persistency
    typedef unsigned int id_type;
    typedef T data_type;

    typedef std::vector<data_type> DataContainer;
    typedef data_type *iterator;
    typedef data_type const *const_iterator;

    typedef data_type value_type;
    typedef id_type key_type;

    inline DetSet() : m_id(0), m_data(nullptr), m_offset(-1), m_size(0) {}
    inline DetSet(id_type i, DataContainer const &idata, size_type ioffset, size_type isize)
        : m_id(i), m_data(&idata), m_offset(ioffset), m_size(isize) {}

    inline DetSet(Container const &icont, typename Container::Item const &item, bool update)
        : m_id(0), m_data(nullptr), m_offset(-1), m_size(0) {
      set(icont, item, update);
    }

    bool isValid() const { return m_offset >= 0; }

    inline data_type &operator[](size_type i) { return data()[i]; }

    inline data_type operator[](size_type i) const { return data()[i]; }

    inline iterator begin() { return data(); }

    inline iterator end() { return data() + m_size; }

    inline const_iterator begin() const { return data(); }

    inline const_iterator end() const { return data() + m_size; }

    int offset() const { return m_offset; }

    inline id_type id() const { return m_id; }

    inline id_type detId() const { return m_id; }

    inline size_type size() const { return m_size; }

    inline bool empty() const { return m_size == 0; }

    template <typename HandleT>
    edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type> makeRefTo(
        HandleT const &handle, const_iterator ci) const {
      return edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>(
          handle.id(), ci, ci - &(container().front()));
    }

    unsigned int makeKeyOf(const_iterator ci) const { return ci - &(container().front()); }

  private:
    //FIXME (it may confuse users as size_type is same type as id_type...)
    inline void set(Container const &icont, typename Container::Item const &item, bool update = true);

    DataContainer const &container() const { return *m_data; }

    data_type const *data() const {
      if (isValid() || !empty())
        assert(m_data);
      return m_data ? (&((*m_data)[m_offset])) : nullptr;
    }

    data_type *data() {
      assert(m_data);
      return const_cast<data_type *>(&((*m_data)[m_offset]));
    }

    id_type m_id;
    DataContainer const *m_data;
    int m_offset;
    size_type m_size;
  };
}  // namespace edmNew

#endif  // DataFormats_Common_DetSet_h
