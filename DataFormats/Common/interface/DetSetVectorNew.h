#ifndef DataFormats_Common_DetSetVectorNew_h
#define DataFormats_Common_DetSetVectorNew_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
// #include "DataFormats/Common/interface/DetSet.h"  // to get det_id_type
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/traits.h"

#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/any.hpp>
#include "boost/shared_ptr.hpp"


#include<vector>

namespace edm { namespace refhelper { template<typename T> struct FindForNewDetSetVector; } }

//FIXME remove New when ready
namespace edmNew {
  typedef uint32_t det_id_type;

  namespace dslv {
    template< typename T> class LazyGetter;
  }

  /* transient component of DetSetVector
   * for pure conviniency of dictioanary declaration
   */
  namespace dstvdetails {
    struct DetSetVectorTrans {
      DetSetVectorTrans(): filling(false){}
      bool filling;
      boost::any getter;

      typedef unsigned int size_type; // for persistency
      typedef unsigned int id_type;

      struct Item {
	Item(id_type i=0, int io=-1, size_type is=0) : id(i), offset(io), size(is){}
	id_type id;
	int offset;
	size_type size;
	bool operator<(Item const &rh) const { return id<rh.id;}
	operator id_type() const { return id;}
      };

    };
    void errorFilling();
    void errorIdExists(det_id_type iid);
    void throw_range(det_id_type iid);
   }

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
  template<typename T>
  class DetSetVector  : private dstvdetails::DetSetVectorTrans {
  public:
    typedef dstvdetails::DetSetVectorTrans Trans;
    typedef Trans::Item Item;
    typedef unsigned int size_type; // for persistency
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
    typedef std::pair<IdIter,DataIter> IterPair;
    typedef typename IdContainer::const_iterator const_IdIter;
    typedef typename std::vector<data_type>::const_iterator const_DataIter;
    typedef std::pair<const_IdIter,const_DataIter> const_IterPair;

    typedef typename edm::refhelper::FindForNewDetSetVector<data_type>  RefFinder;
    
    struct IterHelp {
      typedef DetSet result_type;
      IterHelp() : v(0){}
      IterHelp(DetSetVector<T> const & iv) : v(&iv){}
      
       result_type & operator()(Item const& item) const {
	detset.set(*v,item);
	return detset;
      } 
    private:
      DetSetVector<T> const * v;
      mutable result_type detset;
    };
    
    typedef boost::transform_iterator<IterHelp,const_IdIter> const_iterator;
    typedef std::pair<const_iterator,const_iterator> Range;

    /* fill the lastest inserted DetSet
     */
    class FastFiller {
    public:
      typedef typename DetSetVector<T>::data_type value_type;
      typedef typename DetSetVector<T>::id_type key_type;
      typedef typename DetSetVector<T>::id_type id_type;
      typedef typename DetSetVector<T>::size_type size_type;

      FastFiller(DetSetVector<T> & iv, id_type id, bool isaveEmpty=false) : 
	v(iv), item(v.push_back(id)), saveEmpty(isaveEmpty) {
	if (v.filling) dstvdetails::errorFilling();
	v.filling=true;
      }
      FastFiller(DetSetVector<T> & iv, typename DetSetVector<T>::Item & it, bool isaveEmpty=false) : 
	v(iv), item(it), saveEmpty(isaveEmpty) {
	if (v.filling) dstvdetails::errorFilling();
	v.filling=true;
      }
      ~FastFiller() {
	if (!saveEmpty && item.size==0) {
	  v.pop_back(item.id);
	}
	v.filling=false;
      }
      
      void abort() {
	v.pop_back(item.id);
	saveEmpty=true; // avoid mess in destructor
      }

      void reserve(size_type s) {
	v.m_data.reserve(item.offset+s);
      }

      void resize(size_type s) {
	v.m_data.resize(item.offset+s);
	item.size=s;
      }

      id_type id() const { return item.id;}
      size_type size() const { return item.size;}
      bool empty() const { return item.size==0;}

      data_type & operator[](size_type i) {
	return 	v.m_data[item.offset+i];
      }
      DataIter begin() { return v.m_data.begin()+ item.offset;}
      DataIter end() { return v.m_data.end();}

      void push_back(data_type const & d) {
	v.m_data.push_back(d);
	item.size++;
      }
      data_type & back() { return v.m_data.back();}
      
    private:
      DetSetVector<T> & v;
      typename DetSetVector<T>::Item & item;
      bool saveEmpty;
    };
    friend class FastFiller;

    class FindForDetSetVector : public std::binary_function<const edmNew::DetSetVector<T>&, unsigned int, const T*> {
    public:
        typedef FindForDetSetVector self;
        typename self::result_type operator()(typename self::first_argument_type iContainer, typename self::second_argument_type iIndex) {
            return &(iContainer.m_data[iIndex]);
        }
    };
    friend class FindForDetSetVector;

    explicit DetSetVector(int isubdet=0) :
      m_subdetId(isubdet) {}

    DetSetVector(boost::shared_ptr<dslv::LazyGetter<T> > iGetter, const std::vector<det_id_type>& iDets,
		 int isubdet=0);


    ~DetSetVector() {
      // delete content if T is pointer...
    }
    
    void swap(DetSetVector & rh) {
      std::swap(m_subdetId,rh.m_subdetId);
      std::swap(m_ids,rh.m_ids);
      std::swap(m_data,rh.m_data);
    }
    
    void swap(IdContainer & iic, DataContainer & idc) {
      std::swap(m_ids,iic);
      std::swap(m_data,idc);
    }
    
    void reserve(size_t isize, size_t dsize) {
      m_ids.reserve(isize);
      m_data.reserve(dsize);
    }
    
    void resize(size_t isize, size_t dsize) {
      m_ids.resize(isize);
      m_data.resize(dsize);
    }
    
    // FIXME not sure what the best way to add one cell to cont
    DetSet insert(id_type iid, data_type const * idata, size_type isize) {
      Item & item = addItem(iid,isize);
      m_data.resize(m_data.size()+isize);
      std::copy(idata,idata+isize,m_data.begin()+item.offset);
     return DetSet(*this,item);
    }
    //make space for it
    DetSet insert(id_type iid, size_type isize) {
      Item & item = addItem(iid,isize);
      m_data.resize(m_data.size()+isize);
      return DetSet(*this,item);
    }

    // to be used with a FastFiller
    Item & push_back(id_type iid) {
      return addItem(iid,0);
    }

    // remove last entry (usually only if empty...)
    void pop_back(id_type iid) {
      const_IdIter p = findItem(iid);
      if (p==m_ids.end()) return; //bha!
      // sanity checks...  (shall we throw or assert?)
      if ((*p).size>0&& (*p).offset>-1 && 
	  m_data.size()==(*p).offset+(*p).size)
	m_data.resize((*p).offset);
      m_ids.erase( m_ids.begin()+(p-m_ids.begin()));
    }

  private:

    Item & addItem(id_type iid,  size_type isize) {
      Item it(iid,size_type(m_data.size()),isize);
      IdIter p = std::lower_bound(m_ids.begin(),
				  m_ids.end(),
				  it);
      if (p!=m_ids.end() && !(it<*p)) dstvdetails::errorIdExists(iid);
      return *m_ids.insert(p,it);
    }



  public:


    //---------------------------------------------------------
    
    bool exists(id_type i) const  {
      return  findItem(i)!=m_ids.end(); 
    }
        
    bool isValid(id_type i) const {
      const_IdIter p = findItem(i);
      return p!=m_ids.end() && (*p).offset!=-1;
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
      if (p==m_ids.end()) dstvdetails::throw_range(i);
      return DetSet(*this,*p);
    }
    
    // slow interface
    const_iterator find(id_type i) const {
      const_IdIter p = findItem(i);
      return (p==m_ids.end()) ? end() :
	boost::make_transform_iterator(p,
				       IterHelp(*this));
    }

    // slow interface
    const_IdIter findItem(id_type i) const {
      std::pair<const_IdIter,const_IdIter> p =
	std::equal_range(m_ids.begin(),m_ids.end(),Item(i));
      return (p.first!=p.second) ? p.first : m_ids.end();
    }
    
    const_iterator begin() const {
      return  boost::make_transform_iterator(m_ids.begin(),
					     IterHelp(*this));
    }

    const_iterator end() const {
      return  boost::make_transform_iterator(m_ids.end(),
					     IterHelp(*this));
    }
    

    // return an iterator range (implemented here to avoid dereference of detset)
    template<typename CMP>
    Range equal_range(id_type i, CMP cmp) const {
      std::pair<const_IdIter,const_IdIter> p =
	std::equal_range(m_ids.begin(),m_ids.end(),i,cmp);
      return  Range(boost::make_transform_iterator(p.first,IterHelp(*this)),
		    boost::make_transform_iterator(p.second,IterHelp(*this))
		    );
    }
    
    int subdetId() const { return m_subdetId; }

    bool empty() const { return m_ids.empty();}


    size_type dataSize() const { return m_data.size(); }
    
    size_type size() const { return m_ids.size();}
    
    //FIXME fast interfaces, not consistent with associative nature of container....

    data_type operator()(size_t cell, size_t frame) const {
      return m_data[m_ids[cell].offset+frame];
    }
    
    data_type const * data(size_t cell) const {
      return &m_data[m_ids[cell].offset];
    }
    
    size_type detsetSize(size_t cell) const { return  m_ids[cell].size; }

    id_type id(size_t cell) const {
      return m_ids[cell].id;
    }

    Item const & item(size_t cell) const {
      return m_ids[cell];
    }

    //------------------------------

    // IdContainer const & ids() const { return m_ids;}
    DataContainer const & data() const { return  m_data;}


    void update(Item const & item) const {
      const_cast<self*>(this)->updateImpl(const_cast<Item&>(item));
    }
   
    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:

    void updateImpl(Item & item);
    
  private:
    // subdetector id (as returned by  DetId::subdetId())
    int m_subdetId;
    
    
    IdContainer m_ids;
    DataContainer m_data;
    
  };
  
 namespace dslv {
    template< typename T>
    class LazyGetter {
    public:
      virtual ~LazyGetter() {}
      virtual void fill(typename DetSetVector<T>::FastFiller&) = 0;
    };
  }
  
    

  template<typename T>
  inline DetSetVector<T>::DetSetVector(boost::shared_ptr<dslv::LazyGetter<T> > iGetter, 
				       const std::vector<det_id_type>& iDets,
				       int isubdet):  
    m_subdetId(isubdet) {
    getter=iGetter;

    m_ids.reserve(iDets.size());
    det_id_type sanityCheck = 0;
    for(std::vector<det_id_type>::const_iterator itDetId = iDets.begin(), itDetIdEnd = iDets.end();
	itDetId != itDetIdEnd;
	++itDetId) {
      assert(sanityCheck < *itDetId && "vector of det_id_type was not ordered");
      sanityCheck = *itDetId;
      m_ids.push_back(*itDetId);
    }
  }

  template<typename T>
  inline void DetSetVector<T>::updateImpl(Item & item)  {
    // no getter or already updated
    if (getter.empty() || item.offset!=-1) return;
    item.offset = int(m_data.size());
    FastFiller ff(*this,item);
    (*boost::any_cast<boost::shared_ptr<Getter> >(&getter))->fill(ff);
  }


  template<typename T>
  inline DetSet<T>::DetSet(DetSetVector<T> const & icont,
			   typename DetSetVector<T>::Item const & item ) :
    m_id(0), m_data(0), m_size(0){
    icont.update(item);
    set(icont,item);
  }
  
  
  template<typename T>
  inline void DetSet<T>::set(DetSetVector<T> const & icont,
			     typename Container::Item const & item) {
    icont.update(item);
    m_id=item.id; 
    m_data=&icont.data()[item.offset]; 
    m_size=item.size;
  }
  
}

#include "DataFormats/Common/interface/Ref.h"
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>

//specialize behavior of edm::Ref to get access to the 'Det'
namespace edm {
    /* Reference to an item inside a new DetSetVector ... */
    namespace refhelper {
        template<typename T>
            struct FindTrait<typename edmNew::DetSetVector<T>,T> {
                typedef typename edmNew::DetSetVector<T>::FindForDetSetVector value;
            };
    }
    /* ... as there was one for the original DetSetVector*/

    /* Probably this one is not that useful .... */
    namespace refhelper {
        template<typename T>
            struct FindSetForNewDetSetVector : public std::binary_function<const edmNew::DetSetVector<T>&, unsigned int, edmNew::DetSet<T> > {
                typedef FindSetForNewDetSetVector<T> self;
                typename self::result_type operator()(typename self::first_argument_type iContainer, typename self::second_argument_type iIndex) {
                    return &(iContainer[iIndex]);
                }
            };

        template<typename T>
            struct FindTrait<edmNew::DetSetVector<T>, edmNew::DetSet<T> > {
                typedef FindSetForNewDetSetVector<T> value;
            };
    }
    /* ... implementation is provided, just in case it's needed */
}

namespace edmNew {
   //helper function to make it easier to create a edm::Ref to a new DSV
  template<class HandleT>
  edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>
  makeRefTo(const HandleT& iHandle,
             typename HandleT::element_type::value_type::const_iterator itIter) {
    BOOST_MPL_ASSERT((boost::is_same<typename HandleT::element_type, DetSetVector<typename HandleT::element_type::value_type::value_type> >));
    typename HandleT::element_type::size_type index = (itIter - &*iHandle->data().begin()); 
    return edm::Ref<typename HandleT::element_type,
	       typename HandleT::element_type::value_type::value_type>
	      (iHandle,index);
  }
}


#endif
  
