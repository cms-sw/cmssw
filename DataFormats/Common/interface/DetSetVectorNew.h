#ifndef DataFormats_Common_DetSetVectorNew_h
#define DataFormats_Common_DetSetVectorNew_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/traits.h"

#include <boost/iterator/transform_iterator.hpp>
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <atomic>
#include <memory>
#include <vector>
#include <cassert>

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>

class TestDetSet;

namespace edm {
  namespace refhelper {
    template <typename T>
    struct FindForNewDetSetVector;
  }
}  // namespace edm

//FIXME remove New when ready
namespace edmNew {
  typedef uint32_t det_id_type;

  struct CapacityExaustedException : public cms::Exception {
    CapacityExaustedException() : cms::Exception("Capacity exausted in DetSetVectorNew") {}
  };

  namespace dslv {
    template <typename T>
    class LazyGetter;

  }

  /* transient component of DetSetVector
   * for pure conviniency of dictionary declaration
   */
  namespace dstvdetails {

    void errorFilling();
    void notSafe();
    void errorIdExists(det_id_type iid);
    void throw_range(det_id_type iid);

    struct DetSetVectorTrans {
      typedef unsigned int size_type;  // for persistency
      typedef unsigned int id_type;

      DetSetVectorTrans() : m_filling(false), m_dataSize(0) {}
      DetSetVectorTrans& operator=(const DetSetVectorTrans&) = delete;

      DetSetVectorTrans(const DetSetVectorTrans& rh)
          :  // can't be default because of atomics
            m_filling(false) {
        // better no one is filling...
        assert(rh.m_filling == false);
        m_getter = rh.m_getter;
        m_dataSize.store(rh.m_dataSize.load());
      }

      DetSetVectorTrans(DetSetVectorTrans&& rh)
          :  // can't be default because of atomics
            m_filling(false) {
        // better no one is filling...
        assert(rh.m_filling == false);
        m_getter = std::move(rh.m_getter);
        m_dataSize.store(rh.m_dataSize.exchange(m_dataSize.load()));
      }
      DetSetVectorTrans& operator=(DetSetVectorTrans&& rh) {  // can't be default because of atomics
        // better no one is filling...
        assert(m_filling == false);
        assert(rh.m_filling == false);
        m_getter = std::move(rh.m_getter);
        m_dataSize.store(rh.m_dataSize.exchange(m_dataSize.load()));
        return *this;
      }
      mutable std::atomic<bool> m_filling;
      std::shared_ptr<void> m_getter;
      mutable std::atomic<size_type> m_dataSize;

      void swap(DetSetVectorTrans& rh) {
        // better no one is filling...
        assert(m_filling == false);
        assert(rh.m_filling == false);
        //	std::swap(m_filling,rh.m_filling);
        std::swap(m_getter, rh.m_getter);
        m_dataSize.store(rh.m_dataSize.exchange(m_dataSize.load()));
      }

      struct Item {
        Item(id_type i = 0, int io = -1, size_type is = 0) : id(i), offset(io), size(is) {}

        Item(Item const& rh) noexcept : id(rh.id), offset(int(rh.offset)), size(rh.size) {}
        Item& operator=(Item const& rh) noexcept {
          id = rh.id;
          offset = int(rh.offset);
          size = rh.size;
          return *this;
        }
        Item(Item&& rh) noexcept : id(rh.id), offset(int(rh.offset)), size(rh.size) {}
        Item& operator=(Item&& rh) noexcept {
          id = rh.id;
          offset = int(rh.offset);
          size = rh.size;
          return *this;
        }

        id_type id;
        mutable std::atomic<int> offset;
        bool initialize() const {
          int expected = -1;
          return offset.compare_exchange_strong(expected, -2);
        }
        CMS_THREAD_GUARD(offset) mutable size_type size;

        bool uninitialized() const { return (-1) == offset; }
        bool initializing() const { return (-2) == offset; }
        bool isValid() const { return offset >= 0; }
        bool operator<(Item const& rh) const { return id < rh.id; }
        operator id_type() const { return id; }
      };

      bool ready() const {
        bool expected = false;
        if (!m_filling.compare_exchange_strong(expected, true))
          errorFilling();
        return true;
      }
    };

    inline void throwCapacityExausted() { throw CapacityExaustedException(); }
  }  // namespace dstvdetails

  /** an optitimized container that linearized a "map of vector".
   *  It corresponds to a set of variable size array of T each belonging
   *  to a "Det" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   *
   * although it is sorted internally it is strongly adviced to
   * fill it already sorted....
   *
   */
  template <typename T>
  class DetSetVector : private dstvdetails::DetSetVectorTrans {
  public:
    typedef dstvdetails::DetSetVectorTrans Trans;
    typedef Trans::Item Item;
    typedef unsigned int size_type;  // for persistency
    typedef unsigned int id_type;
    typedef T data_type;
    typedef edmNew::DetSetVector<T> self;
    typedef edmNew::DetSet<T> DetSet;
    typedef dslv::LazyGetter<T> Getter;
    // FIXME not sure make sense....
    typedef DetSet value_type;
    typedef id_type key_type;

    typedef std::vector<Item> IdContainer;
    typedef std::vector<data_type> DataContainer;
    typedef typename IdContainer::iterator IdIter;
    typedef typename std::vector<data_type>::iterator DataIter;
    typedef std::pair<IdIter, DataIter> IterPair;
    typedef typename IdContainer::const_iterator const_IdIter;
    typedef typename std::vector<data_type>::const_iterator const_DataIter;
    typedef std::pair<const_IdIter, const_DataIter> const_IterPair;

    typedef typename edm::refhelper::FindForNewDetSetVector<data_type> RefFinder;

    struct IterHelp {
      typedef DetSet result_type;
      //      IterHelp() : v(0),update(true){}
      IterHelp() : m_v(nullptr), m_update(false) {}
      IterHelp(DetSetVector<T> const& iv, bool iup) : m_v(&iv), m_update(iup) {}

      result_type operator()(Item const& item) const { return result_type(*m_v, item, m_update); }

    private:
      DetSetVector<T> const* m_v;
      bool m_update;
    };

    typedef boost::transform_iterator<IterHelp, const_IdIter> const_iterator;
    typedef std::pair<const_iterator, const_iterator> Range;

    /* fill the lastest inserted DetSet
     */
    class FastFiller {
    public:
      typedef typename DetSetVector<T>::data_type value_type;
      typedef typename DetSetVector<T>::id_type key_type;
      typedef typename DetSetVector<T>::id_type id_type;
      typedef typename DetSetVector<T>::size_type size_type;

      // here just to make the compiler happy
      static DetSetVector<T>::Item& dummy() {
        assert(false);
        static DetSetVector<T>::Item d;
        return d;
      }

      FastFiller(DetSetVector<T>& iv, id_type id, bool isaveEmpty = false)
          : m_v(iv), m_item(m_v.ready() ? m_v.push_back(id) : dummy()), m_saveEmpty(isaveEmpty) {
        if (m_v.onDemand())
          dstvdetails::notSafe();
      }

      FastFiller(DetSetVector<T>& iv, typename DetSetVector<T>::Item& it, bool isaveEmpty = false)
          : m_v(iv), m_item(it), m_saveEmpty(isaveEmpty) {
        if (m_v.onDemand())
          dstvdetails::notSafe();
        if (m_v.ready())
          m_item.offset = int(m_v.m_data.size());
      }
      ~FastFiller() {
        if (!m_saveEmpty && m_item.size == 0) {
          m_v.pop_back(m_item.id);
        }
        assert(m_v.m_filling == true);
        m_v.m_filling = false;
      }

      void abort() {
        m_v.pop_back(m_item.id);
        m_saveEmpty = true;  // avoid mess in destructor
      }

      void checkCapacityExausted() const {
        if (m_v.onDemand() && m_v.m_data.size() == m_v.m_data.capacity())
          dstvdetails::throwCapacityExausted();
      }

      void checkCapacityExausted(size_type s) const {
        if (m_v.onDemand() && m_v.m_data.size() + s > m_v.m_data.capacity())
          dstvdetails::throwCapacityExausted();
      }

      void reserve(size_type s) {
        if (m_item.offset + s <= m_v.m_data.capacity())
          return;
        if (m_v.onDemand())
          dstvdetails::throwCapacityExausted();
        m_v.m_data.reserve(m_item.offset + s);
      }

      void resize(size_type s) {
        checkCapacityExausted(s);
        m_v.m_data.resize(m_item.offset + s);
        m_v.m_dataSize = m_v.m_data.size();
        m_item.size = s;
      }

      id_type id() const { return m_item.id; }
      size_type size() const { return m_item.size; }
      bool empty() const { return m_item.size == 0; }

      data_type& operator[](size_type i) { return m_v.m_data[m_item.offset + i]; }
      DataIter begin() { return m_v.m_data.begin() + m_item.offset; }
      DataIter end() { return begin() + size(); }

      template <typename... Args>
      void emplace_back(Args&&... args) {
        checkCapacityExausted();
        m_v.m_data.emplace_back(args...);
        ++m_v.m_dataSize;
        m_item.size++;
      }

      void push_back(data_type const& d) {
        checkCapacityExausted();
        m_v.m_data.push_back(d);
        ++m_v.m_dataSize;
        m_item.size++;
      }

      void push_back(data_type&& d) {
        checkCapacityExausted();
        m_v.m_data.push_back(std::move(d));
        ++m_v.m_dataSize;
        m_item.size++;
      }

      data_type& back() { return m_v.m_data.back(); }

    private:
      //for testing
      friend class ::TestDetSet;

      DetSetVector<T>& m_v;
      typename DetSetVector<T>::Item& m_item;
      bool m_saveEmpty;
    };

    /* fill on demand a given  DetSet
     */
    class TSFastFiller {
    public:
      typedef typename DetSetVector<T>::data_type value_type;
      typedef typename DetSetVector<T>::id_type key_type;
      typedef typename DetSetVector<T>::id_type id_type;
      typedef typename DetSetVector<T>::size_type size_type;

      // here just to make the compiler happy
      static DetSetVector<T>::Item const& dummy() {
        assert(false);
        static DetSetVector<T>::Item const d;
        return d;
      }
      // this constructor is not supposed to be used in Concurrent mode
      TSFastFiller(DetSetVector<T>& iv, id_type id) : m_v(iv), m_item(m_v.ready() ? iv.push_back(id) : dummy()) {
        assert(m_v.m_filling == true);
        m_v.m_filling = false;
      }

      TSFastFiller(DetSetVector<T> const& iv, typename DetSetVector<T>::Item const& it) : m_v(iv), m_item(it) {}
      ~TSFastFiller() {
        bool expected = false;
        while (!m_v.m_filling.compare_exchange_weak(expected, true)) {
          expected = false;
          nanosleep(nullptr, nullptr);
        }
        int offset = m_v.m_data.size();
        if (m_v.onDemand() && full()) {
          m_v.m_filling = false;
          dstvdetails::throwCapacityExausted();
        }
        std::move(m_lv.begin(), m_lv.end(), std::back_inserter(m_v.m_data));
        m_item.size = m_lv.size();
        m_item.offset = offset;

        m_v.m_dataSize = m_v.m_data.size();
        assert(m_v.m_filling == true);
        m_v.m_filling = false;
      }

      bool full() const {
        int offset = m_v.m_dataSize;
        return m_v.m_data.capacity() < offset + m_lv.size();
      }

      void abort() { m_lv.clear(); }

      void reserve(size_type s) { m_lv.reserve(s); }

      void resize(size_type s) { m_lv.resize(s); }

      id_type id() const { return m_item.id; }
      size_type size() const { return m_lv.size(); }
      bool empty() const { return m_lv.empty(); }

      data_type& operator[](size_type i) { return m_lv[i]; }
      DataIter begin() { return m_lv.begin(); }
      DataIter end() { return m_lv.end(); }

      template <typename... Args>
      void emplace_back(Args&&... args) {
        m_lv.emplace_back(args...);
      }

      void push_back(data_type const& d) { m_lv.push_back(d); }
      void push_back(data_type&& d) { m_lv.push_back(std::move(d)); }

      data_type& back() { return m_lv.back(); }

    private:
      //for testing
      friend class ::TestDetSet;

      std::vector<T> m_lv;
      DetSetVector<T> const& m_v;
      typename DetSetVector<T>::Item const& m_item;
    };

    friend class FastFiller;
    friend class TSFastFiller;
    friend class edmNew::DetSet<T>;

    class FindForDetSetVector {
    public:
      using first_argument_type = const edmNew::DetSetVector<T>&;
      using second_argument_type = unsigned int;
      using result_type = const T*;

      result_type operator()(first_argument_type iContainer, second_argument_type iIndex) {
        bool expected = false;
        while (!iContainer.m_filling.compare_exchange_weak(expected, true, std::memory_order_acq_rel)) {
          expected = false;
          nanosleep(nullptr, nullptr);
        }
        result_type item = &(iContainer.m_data[iIndex]);
        assert(iContainer.m_filling == true);
        iContainer.m_filling = false;
        return item;
      }
    };
    friend class FindForDetSetVector;

    explicit DetSetVector(int isubdet = 0) : m_subdetId(isubdet) {}

    DetSetVector(std::shared_ptr<dslv::LazyGetter<T>> iGetter, const std::vector<det_id_type>& iDets, int isubdet = 0);

    ~DetSetVector() {
      // delete content if T is pointer...
    }

    // default or delete is the same...
    DetSetVector& operator=(const DetSetVector&) = delete;
    // Implement copy constructor because of a (possibly temporary)
    // need in heterogeneous framework prototyping. In general this
    // class is still supposed to be non-copyable, so to prevent
    // accidental copying the assignment operator is left deleted.
    DetSetVector(const DetSetVector&) = default;
    DetSetVector(DetSetVector&&) = default;
    DetSetVector& operator=(DetSetVector&&) = default;

    bool onDemand() const { return static_cast<bool>(m_getter); }

    void swap(DetSetVector& rh) {
      DetSetVectorTrans::swap(rh);
      std::swap(m_subdetId, rh.m_subdetId);
      std::swap(m_ids, rh.m_ids);
      std::swap(m_data, rh.m_data);
    }

    void swap(IdContainer& iic, DataContainer& idc) {
      std::swap(m_ids, iic);
      std::swap(m_data, idc);
    }

    void reserve(size_t isize, size_t dsize) {
      m_ids.reserve(isize);
      m_data.reserve(dsize);
    }

    void shrink_to_fit() {
      clean();
      m_ids.shrink_to_fit();
      m_data.shrink_to_fit();
    }

    void resize(size_t isize, size_t dsize) {
      m_ids.resize(isize);
      m_data.resize(dsize);
      m_dataSize = m_data.size();
    }

    void clean() {
      m_ids.erase(std::remove_if(m_ids.begin(), m_ids.end(), [](Item const& m) { return 0 == m.size; }), m_ids.end());
    }

    // FIXME not sure what the best way to add one cell to cont
    DetSet insert(id_type iid, data_type const* idata, size_type isize) {
      Item& item = addItem(iid, isize);
      m_data.resize(m_data.size() + isize);
      std::copy(idata, idata + isize, m_data.begin() + item.offset);
      m_dataSize = m_data.size();
      return DetSet(*this, item, false);
    }
    //make space for it
    DetSet insert(id_type iid, size_type isize) {
      Item& item = addItem(iid, isize);
      m_data.resize(m_data.size() + isize);
      m_dataSize = m_data.size();
      return DetSet(*this, item, false);
    }

    // to be used with a FastFiller
    Item& push_back(id_type iid) { return addItem(iid, 0); }

    // remove last entry (usually only if empty...)
    void pop_back(id_type iid) {
      const_IdIter p = findItem(iid);
      if (p == m_ids.end())
        return;  //bha!
      // sanity checks...  (shall we throw or assert?)
      if ((*p).isValid() && (*p).size > 0 && m_data.size() == (*p).offset + (*p).size) {
        m_data.resize((*p).offset);
        m_dataSize = m_data.size();
      }
      m_ids.erase(m_ids.begin() + (p - m_ids.begin()));
    }

  private:
    Item& addItem(id_type iid, size_type isize) {
      Item it(iid, size_type(m_data.size()), isize);
      IdIter p = std::lower_bound(m_ids.begin(), m_ids.end(), it);
      if (p != m_ids.end() && !(it < *p))
        dstvdetails::errorIdExists(iid);
      return *m_ids.insert(p, std::move(it));
    }

  public:
    //---------------------------------------------------------

    bool exists(id_type i) const { return findItem(i) != m_ids.end(); }

    bool isValid(id_type i) const {
      const_IdIter p = findItem(i);
      return p != m_ids.end() && (*p).isValid();
    }

    /*
    DetSet operator[](id_type i) {
      const_IdIter p = findItem(i);
      if (p==m_ids.end()) what???
      return DetSet(*this,p-m_ids.begin());
    }
    */

    DetSet operator[](id_type i) const {
      const_IdIter p = findItem(i);
      if (p == m_ids.end())
        dstvdetails::throw_range(i);
      return DetSet(*this, *p, true);
    }

    // slow interface
    //    const_iterator find(id_type i, bool update=true) const {
    const_iterator find(id_type i, bool update = false) const {
      const_IdIter p = findItem(i);
      return (p == m_ids.end()) ? end() : boost::make_transform_iterator(p, IterHelp(*this, update));
    }

    // slow interface
    const_IdIter findItem(id_type i) const {
      std::pair<const_IdIter, const_IdIter> p = std::equal_range(m_ids.begin(), m_ids.end(), Item(i));
      return (p.first != p.second) ? p.first : m_ids.end();
    }

    //    const_iterator begin(bool update=true) const {
    const_iterator begin(bool update = false) const {
      return boost::make_transform_iterator(m_ids.begin(), IterHelp(*this, update));
    }

    //    const_iterator end(bool update=true) const {
    const_iterator end(bool update = false) const {
      return boost::make_transform_iterator(m_ids.end(), IterHelp(*this, update));
    }

    // return an iterator range (implemented here to avoid dereference of detset)
    template <typename CMP>
    //    Range equal_range(id_type i, CMP cmp, bool update=true) const {
    Range equal_range(id_type i, CMP cmp, bool update = false) const {
      std::pair<const_IdIter, const_IdIter> p = std::equal_range(m_ids.begin(), m_ids.end(), i, cmp);
      return Range(boost::make_transform_iterator(p.first, IterHelp(*this, update)),
                   boost::make_transform_iterator(p.second, IterHelp(*this, update)));
    }

    int subdetId() const { return m_subdetId; }

    bool empty() const { return m_ids.empty(); }

    size_type dataSize() const { return onDemand() ? size_type(m_dataSize) : size_type(m_data.size()); }

    size_type size() const { return m_ids.size(); }

    //FIXME fast interfaces, not consistent with associative nature of container....

    data_type operator()(size_t cell, size_t frame) const { return m_data[m_ids[cell].offset + frame]; }

    data_type const* data(size_t cell) const { return &m_data[m_ids[cell].offset]; }

    size_type detsetSize(size_t cell) const { return m_ids[cell].size; }

    id_type id(size_t cell) const { return m_ids[cell].id; }

    Item const& item(size_t cell) const { return m_ids[cell]; }

    //------------------------------

    IdContainer const& ids() const { return m_ids; }
    DataContainer const& data() const { return m_data; }

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    //for testing
    friend class ::TestDetSet;

    void update(Item const& item) const;

    // subdetector id (as returned by  DetId::subdetId())
    int m_subdetId;

    // Workaround for ROOT 6 bug.
    // ROOT6 has a problem with this IdContainer typedef
    //IdContainer m_ids;
    std::vector<Trans::Item> m_ids;
    CMS_THREAD_GUARD(dstvdetails::DetSetVectorTrans::m_filling) mutable DataContainer m_data;
  };

  namespace dslv {
    template <typename T>
    class LazyGetter {
    public:
      virtual ~LazyGetter() {}
      virtual void fill(typename DetSetVector<T>::TSFastFiller&) = 0;
    };
  }  // namespace dslv

  template <typename T>
  inline DetSetVector<T>::DetSetVector(std::shared_ptr<Getter> iGetter,
                                       const std::vector<det_id_type>& iDets,
                                       int isubdet)
      : m_subdetId(isubdet) {
    m_getter = iGetter;

    m_ids.reserve(iDets.size());
    det_id_type sanityCheck = 0;
    for (std::vector<det_id_type>::const_iterator itDetId = iDets.begin(), itDetIdEnd = iDets.end();
         itDetId != itDetIdEnd;
         ++itDetId) {
      assert(sanityCheck < *itDetId && "vector of det_id_type was not ordered");
      sanityCheck = *itDetId;
      m_ids.push_back(*itDetId);
    }
  }

  template <typename T>
  inline void DetSetVector<T>::update(const Item& item) const {
    // no m_getter or already updated
    if (!m_getter) {
      assert(item.isValid());
      return;
    }
    if (item.initialize()) {
      assert(item.initializing());
      {
        TSFastFiller ff(*this, item);
        static_cast<Getter*>(m_getter.get())->fill(ff);
      }
      assert(item.isValid());
    }
  }

  template <typename T>
  inline void DetSet<T>::set(DetSetVector<T> const& icont, typename Container::Item const& item, bool update) {
    // if an item is being updated we wait
    if (update)
      icont.update(item);
    while (item.initializing())
      nanosleep(nullptr, nullptr);
    m_data = &icont.data();
    m_id = item.id;
    m_offset = item.offset;
    m_size = item.size;
  }
}  // namespace edmNew

#include "DataFormats/Common/interface/Ref.h"
#include <type_traits>

//specialize behavior of edm::Ref to get access to the 'Det'
namespace edm {
  /* Reference to an item inside a new DetSetVector ... */
  namespace refhelper {
    template <typename T>
    struct FindTrait<typename edmNew::DetSetVector<T>, T> {
      typedef typename edmNew::DetSetVector<T>::FindForDetSetVector value;
    };
  }  // namespace refhelper
  /* ... as there was one for the original DetSetVector*/

  /* Probably this one is not that useful .... */
  namespace refhelper {
    template <typename T>
    struct FindSetForNewDetSetVector {
      using first_argument_type = const edmNew::DetSetVector<T>&;
      using second_argument_type = unsigned int;
      using result_type = edmNew::DetSet<T>;

      result_type operator()(first_argument_type iContainer, second_argument_type iIndex) {
        return &(iContainer[iIndex]);
      }
    };

    template <typename T>
    struct FindTrait<edmNew::DetSetVector<T>, edmNew::DetSet<T>> {
      typedef FindSetForNewDetSetVector<T> value;
    };
  }  // namespace refhelper
     /* ... implementation is provided, just in case it's needed */
}  // namespace edm

namespace edmNew {
  //helper function to make it easier to create a edm::Ref to a new DSV
  template <class HandleT>
  // inline
  edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type> makeRefTo(
      const HandleT& iHandle, typename HandleT::element_type::value_type::const_iterator itIter) {
    static_assert(std::is_same<typename HandleT::element_type,
                               DetSetVector<typename HandleT::element_type::value_type::value_type>>::value,
                  "Handle and DetSetVector do not have compatible types.");
    auto index = itIter - &iHandle->data().front();
    return edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>(
        iHandle.id(), &(*itIter), index);
  }
}  // namespace edmNew

#include "DataFormats/Common/interface/ContainerMaskTraits.h"

namespace edm {
  template <typename T>
  class ContainerMaskTraits<edmNew::DetSetVector<T>> {
  public:
    typedef T value_type;

    static size_t size(const edmNew::DetSetVector<T>* iContainer) { return iContainer->dataSize(); }
    static unsigned int indexFor(const value_type* iElement, const edmNew::DetSetVector<T>* iContainer) {
      return iElement - &(iContainer->data().front());
    }
  };
}  // namespace edm

// Thinning support
#include "DataFormats/Common/interface/fillCollectionForThinning.h"
namespace edm::detail {
  template <typename T>
  struct ElementType<edmNew::DetSetVector<T>> {
    using type = typename edmNew::DetSetVector<T>::data_type;
  };
}  // namespace edm::detail
namespace edmNew {
  template <typename T, typename Selector>
  void fillCollectionForThinning(edmNew::DetSet<T> const& detset,
                                 Selector& selector,
                                 unsigned int& iIndex,
                                 edmNew::DetSetVector<T>& output,
                                 edm::ThinnedAssociation& association) {
    typename edmNew::DetSetVector<T>::FastFiller ff(output, detset.detId());
    for (auto iter = detset.begin(), end = detset.end(); iter != end; ++iter, ++iIndex) {
      edm::detail::fillCollectionForThinning(*iter, selector, iIndex, ff, association);
    }
    if (detset.begin() != detset.end()) {
      // need to decrease the global index by one because the outer loop will increase it
      --iIndex;
    }
  }
}  // namespace edmNew

#endif
