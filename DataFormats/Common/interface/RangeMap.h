#ifndef Common_RangeMap_h
#define Common_RangeMap_h
#include <map>
#include <vector>
#include <functional>
#include <ext/functional>
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/traits.h"

// $Id: RangeMap.h,v 1.10 2006/02/17 10:30:00 llista Exp $
namespace edm {
  
  template<typename ID, typename C, typename P>
    class RangeMap {
    public:
    typedef typename C::value_type value_type;
    typedef typename C::const_iterator const_iterator;
    typedef std::pair<typename C::size_type, typename C::size_type> pairType;
    typedef std::map<ID, pairType> mapType;
    typedef std::pair<const_iterator, const_iterator> range;
          
    template<typename COMP> 
    range get(ID id, COMP comparator){
      using __gnu_cxx::select1st;
      std::pair<typename mapType::const_iterator,
        typename mapType::const_iterator> r =
        std::equal_range( map_.begin(), map_.end(), id,
                          std::compose2( comparator,
                                         select1st<pairType>(),
                                         select1st<pairType>() ) );
      const_iterator begin, end;

      if ((r.first) == map_.end()){
	begin = end = collection_.end();
      }else{
	begin = collection_.begin() + (r.first)->second.first;
	end = collection_.begin() + (r.second)->second.second;
      }
      return  make_pair(begin,end);
    }

    RangeMap() { }
    range get( ID id ) const {
      const_iterator begin, end;
      typename mapType::const_iterator i = map_.find( id );
      if ( i != map_.end() ) { 
	begin = collection_.begin() + i->second.first;
	end = collection_.begin() + i->second.second;
      } else {
	begin = end = collection_.end();
      }
      return std::make_pair( begin, end );
    }

    template<typename CI>
      void put( ID id, CI begin, CI end ) {
      typename mapType::const_iterator i = map_.find( id );
      if( i != map_.end() ) 
      	throw cms::Exception( "Error" ) << "trying to insert duplicate entry";
      assert( i == map_.end() );
      pairType & p = map_[ id ];
      p.first = collection_.size();
      for( CI i = begin; i != end; ++ i )
	collection_.push_back( P::clone( * i ) );
      p.second = collection_.size();
    }
    size_t size() { return collection_.size(); }
    typename C::const_iterator begin() const { return collection_.begin(); }
    typename C::const_iterator end() const { return collection_.end(); }
    
    struct id_iterator {
      typedef ID value_type;
      typedef ID * pointer;
      typedef ID & reference;
      typedef ptrdiff_t difference_type;
      typedef typename mapType::const_iterator::iterator_category iterator_category;
      typedef typename mapType::const_iterator const_iterator;
      id_iterator() { }
      id_iterator( const_iterator o ) : i( o ) { }
      id_iterator & operator=( const id_iterator & it ) { i = it.i; return *this; }
      id_iterator& operator++() { ++i; return *this; }
      id_iterator operator++( int ) { id_iterator ci = *this; ++i; return ci; }
      id_iterator& operator--() { --i; return *this; }
      id_iterator operator--( int ) { id_iterator ci = *this; --i; return ci; }
      bool operator==( const id_iterator& ci ) const { return i == ci.i; }
      bool operator!=( const id_iterator& ci ) const { return i != ci.i; }
      const ID operator * () const { return i->first; }      
      private:
      const_iterator i;
    };

    void post_insert(){
      // sorts the container via ID
      C tmp;
      for (typename mapType::iterator it = map_.begin(); it != map_.end(); it ++) {   
	range r = get( (*it).first );
	typename C::size_type begIt = tmp.size();
	for( const_iterator i = r.first; i != r.second; ++i )
	  tmp.push_back( P::clone( *i ) );
	typename C::size_type endIt = tmp.size();
	it->second = std::make_pair( begIt, endIt );
      }
      collection_ = tmp;
    }

    id_iterator id_begin() const { return id_iterator( map_.begin() ); }
    id_iterator id_end() const { return id_iterator( map_.end() ); }
    size_t id_size() const { return map_.size(); }
    std::vector<ID> ids() const {
      std::vector<ID> temp( id_size() );
      std::copy( id_begin(), id_end(), temp.begin() );
      return temp;
    }

  private:
    C collection_;
    mapType map_;
  };
  
  template<typename  ID, typename C, typename P > 
  struct edm::has_postinsert_trait<edm::RangeMap<ID,C,P> >  { 
    static bool const value = true; 
  }; 
  
}

#endif
