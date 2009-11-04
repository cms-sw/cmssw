#ifndef DataFormats_Common_RangeMap_h
#define DataFormats_Common_RangeMap_h
/* \class edm::RangeMap
 *
 * Generic container storing objects arranged according 
 * to a specified identifier. 
 *
 * The data content can be fetched either via
 * an iterator, or specifying user-defined identifier 
 * match criteria.
 *
 * The template parameters are:
 * - ID: identifier type
 * - C : underlying collection used to 
 * - P : policy to perform object cloning
 *
 * \author Tommaso Boccali, Luca Lista INFN
 *
 * \version $Revision: 1.36 $
 *
 * $Id: RangeMap.h,v 1.36 2008/07/23 22:50:16 wmtan Exp $
 *
 */
#include <map>
#include <vector>
#include <ext/functional>
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/CloneTrait.h"

namespace edm {
  
  template<typename ID, typename C, typename P = typename clonehelper::CloneTrait<C>::type >
  class RangeMap {
  public:
    /// contained object type
    typedef typename C::value_type value_type;
    /// collection size type
    typedef typename C::size_type size_type;
    /// reference type
    typedef typename C::reference reference;
    /// pointer type
    typedef typename C::pointer pointer;
    /// constant access iterator type
    typedef typename C::const_iterator const_iterator;
    /// index range
    //use unsigned int rather than C::size_type in order to avoid porting problems
    typedef std::pair<unsigned int, unsigned int> pairType;
    /// map of identifier to index range
    typedef std::map<ID, pairType> mapType;
    /// iterator range
    typedef std::pair<const_iterator, const_iterator> range;
    
  private:
    /// comparator helper class
    template<typename CMP> 
    struct comp {
      comp(const CMP  c) : cmp(c) { }
      bool operator()(ID id, const typename mapType::value_type & p) {
	return cmp(id, p.first);
      }
      bool operator()(const typename mapType::value_type & p, ID id) {
	return cmp(p.first, id);
      }
    private:
      CMP cmp;
      
    };

  public:
    /// default constructor
    RangeMap() { }
    /// get range of objects matching a specified identifier with a specified comparator.
    /// <b>WARNING</b>: the comparator has to be written 
    /// in such a way that the std::equal_range 
    /// function returns a meaningful range.
    /// Not properly written comparators may return
    /// an unpredictable range. It is recommended
    /// to use only comparators provided with CMSSW release.
    template<typename CMP> 
    range get(ID id, CMP comparator) const {
      using namespace __gnu_cxx;
      std::pair<typename mapType::const_iterator,
	typename mapType::const_iterator> r =
        std::equal_range(map_.begin(), map_.end(), id, comp<CMP>(comparator));
      const_iterator begin, end;
      if ((r.first) == map_.end()){
	begin = end = collection_.end();
	return  std::make_pair(begin,end);
      } else {
	begin = collection_.begin() + (r.first)->second.first;
      }
      if ((r.second) == map_.end()){
	end = collection_.end();
      }else{
	end = collection_.begin() + (r.second)->second.first;
      }
      return  std::make_pair(begin,end);
    }
    /// get range of objects matching a specified identifier with a specified comparator.
    template<typename CMP> 
    range get(std::pair<ID, CMP> p) const {
      return get(p.first, p.second); 
    }
    /// get a range of objects with specified identifier
    range get(ID id) const {
      const_iterator begin, end;
      typename mapType::const_iterator i = map_.find(id);
      if (i != map_.end()) { 
	begin = collection_.begin() + i->second.first;
	end = collection_.begin() + i->second.second;
      } else {
	begin = end = collection_.end();
      }
      return std::make_pair(begin, end);
    }
    /// insert an object range with specified identifier
    template<typename CI>
    void put(ID id, CI begin, CI end) {
      typename mapType::const_iterator i = map_.find(id);
      if(i != map_.end()) {
      	throw Exception(errors::LogicError, "trying to insert duplicate entry");
      }
      assert(i == map_.end());
      pairType & p = map_[ id ];
      p.first = collection_.size();
      for(CI i = begin; i != end; ++i)
	collection_.push_back(P::clone(*i));
      p.second = collection_.size();
    }
    /// return number of contained object
    size_t size() const { return collection_.size(); }
    /// first collection iterator 
    typename C::const_iterator begin() const { return collection_.begin(); }
    /// last collection iterator 
    typename C::const_iterator end() const { return collection_.end(); }
    /// identifier iterator 
    struct id_iterator {
      typedef ID value_type;
      typedef ID * pointer;
      typedef ID & reference;
      typedef ptrdiff_t difference_type;
      typedef typename mapType::const_iterator::iterator_category iterator_category;
      typedef typename mapType::const_iterator const_iterator;
      id_iterator() { }
      id_iterator(const_iterator o) : i(o) { }
      id_iterator & operator=(const id_iterator & it) { i = it.i; return *this; }
      id_iterator& operator++() { ++i; return *this; }
      id_iterator operator++(int) { id_iterator ci = *this; ++i; return ci; }
      id_iterator& operator--() { --i; return *this; }
      id_iterator operator--(int) { id_iterator ci = *this; --i; return ci; }
      bool operator==(const id_iterator& ci) const { return i == ci.i; }
      bool operator!=(const id_iterator& ci) const { return i != ci.i; }
      const ID operator * () const { return i->first; }      
    private:
      const_iterator i;
    };
    /// perfor post insert action
    void post_insert() {
      // sorts the container via ID
      C tmp;
      for (typename mapType::iterator it = map_.begin(), itEnd = map_.end(); it != itEnd; it ++) {   
	range r = get((*it).first);
	//do cast to acknowledge that we may be going from a larger type to a smaller type but we are OK
	unsigned int begIt = static_cast<unsigned int>(tmp.size());
	for(const_iterator i = r.first; i != r.second; ++i)
	  tmp.push_back(P::clone(*i));
	unsigned int endIt = static_cast<unsigned int>(tmp.size());
	it->second = pairType(begIt, endIt);
      }
      collection_ = tmp;
    }
    /// first identifier iterator 
    id_iterator id_begin() const { return id_iterator(map_.begin()); }
    /// last identifier iterator 
    id_iterator id_end() const { return id_iterator(map_.end()); }
    /// number of contained identifiers 
    size_t id_size() const { return map_.size(); }
    /// indentifier vector
    std::vector<ID> ids() const {
      std::vector<ID> temp(id_size());
      std::copy(id_begin(), id_end(), temp.begin());
      return temp;
    }
    /// direct access to an object in the collection
    reference operator[](size_type i) { return collection_[ i ]; }

    /// swap member function
    void swap(RangeMap<ID, C, P> & other);

    /// copy assignment
    RangeMap& operator=(RangeMap const& rhs);

  private:
    /// stored collection
    C collection_;
    /// identifier map
    mapType map_;
  };

  template <typename ID, typename C, typename P>
  inline
  void
  RangeMap<ID, C, P>::swap(RangeMap<ID, C, P> & other) {
    collection_.swap(other.collection_);
    map_.swap(other.map_);
  }

  template <typename ID, typename C, typename P>
  inline
  RangeMap<ID, C, P>&
  RangeMap<ID, C, P>::operator=(RangeMap<ID, C, P> const& rhs) {
    RangeMap<ID, C, P> temp(rhs);
    this->swap(temp);
    return *this;
  }

  // free swap function
  template <typename ID, typename C, typename P>
  inline
  void
  swap(RangeMap<ID, C, P> & a, RangeMap<ID, C, P> & b) {
    a.swap(b);
  }

}

#endif
