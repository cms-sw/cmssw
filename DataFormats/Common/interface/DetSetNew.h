#ifndef DataFormats_Common_DetSetNew_h
#define DataFormats_Common_DetSetNew_h

#include "DataFormats/Common/interface/Ref.h"
#include <type_traits>
#include <vector>
#include <cassert>

namespace edmNew {
  //  FIXME move it elsewhere....
  using det_id_type = unsigned int;

  template <typename T>
  class DetSetVector;

  template <typename T, bool IsConst>
  class DetSetBase;

  template <typename T>
  using MutableDetSet = DetSetBase<T, false>;

  template <typename T>
  using DetSet = DetSetBase<T, true>;

  /* a proxy to a variable size array of T belonging to
   * a "channel" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   * 
   */
  template <typename T, bool IsConst>
  class DetSetBase {
  public:
    using Container = DetSetVector<T>;
    using ContainerReference = std::conditional_t<IsConst, Container const&, Container&>;
    using Item = typename Container::Item;
    using size_type = unsigned int;  // for persistency
    using id_type = unsigned int;
    using data_type = T;

    using DataContainer = std::vector<data_type>;
    using DataContainerReference = std::conditional_t<IsConst, DataContainer const&, DataContainer&>;
    using iterator = std::conditional_t<IsConst, data_type const*, data_type*>;
    using const_iterator = data_type const*;
    using data_reference = std::conditional_t<IsConst, data_type const&, data_type&>;
    using value_type = data_type;
    using key_type = id_type;

    inline DetSetBase() : m_id(0), m_data(nullptr), m_offset(-1), m_size(0) {}
    inline DetSetBase(id_type i, DataContainerReference idata, size_type ioffset, size_type isize)
        : m_id(i), m_data(&idata), m_offset(ioffset), m_size(isize) {}

    inline DetSetBase(ContainerReference icont, Item const& item, bool update)
        : m_id(0), m_data(nullptr), m_offset(-1), m_size(0) {
      init(icont, item, update);
    }

    //Allow a  DetSet to be constructed from a MutableDetSet
    template <bool = IsConst>
      requires IsConst
    inline DetSetBase(MutableDetSet<T> const& other)
        : m_id(other.m_id), m_data(other.m_data), m_offset(other.m_offset), m_size(other.m_size) {}

    bool isValid() const { return m_offset >= 0; }

    inline data_reference operator[](size_type i) { return data()[i]; }

    inline data_type const& operator[](size_type i) const { return data()[i]; }

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
        HandleT const& handle, const_iterator ci) const {
      return edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>(
          handle.id(), ci, ci - &(container().front()));
    }

    unsigned int makeKeyOf(const_iterator ci) const { return ci - &(container().front()); }

  private:
    friend class DetSetBase<T, true>;

    void init(ContainerReference icont, Item const& item, bool update);

    DataContainer const& container() const { return *m_data; }

    iterator data() const {
      if (isValid() || !empty())
        assert(m_data);
      return m_data ? &((*m_data)[m_offset]) : nullptr;
    }

    id_type m_id;
    std::conditional_t<IsConst, DataContainer const*, DataContainer*> m_data;
    int m_offset;
    size_type m_size;
  };
}  // namespace edmNew

#endif  // DataFormats_Common_DetSet_h
