#ifndef Common_RangeMap_h
#define Common_RangeMap_h
#include <map>
#include "FWCore/Utilities/interface/Exception.h"
// $Id$
namespace edm {

  template<typename ID, typename C, typename P>
  class RangeMap {
  public:
    typedef typename C::value_type value_type;
    typedef typename C::const_iterator const_iterator;
    typedef std::map<ID, typename C::size_type> map;
    typedef std::pair<const_iterator, const_iterator> range;
    RangeMap() { }
    range get( ID id ) const {
      const_iterator begin, end;
      typename map::const_iterator i = map_.find( id );
      if ( i != map_.end() ) { 
	begin = collection_.begin() + i->second;
	end = ( ++ i != map_.end() ) ? 
	  collection_.begin() + i->second :
	  collection_.end();
      } else {
	begin = end = collection_.end();
      }
      return std::make_pair( begin, end );
    }
    template<typename CI>
    void put( ID id, CI begin, CI end ) {
      typename map::const_iterator i = map_.find( id );
      if( i != map_.end() ) 
      	throw cms::Exception( "Error" ) << "trying to insert duplicate entry";
      assert( i == map_.end() );
      map_[ id ] = collection_.size();
      for( CI i = begin; i != end; ++ i )
	collection_.push_back( P::clone( * i ) );
    }
    size_t size() { return collection_.size(); }
    typename C::const_iterator begin() const { return collection_.begin(); }
    typename C::const_iterator end() const { return collection_.end(); }
    struct id_iterator {
      typedef ID value_type;
      typedef ID * pointer;
      typedef ID & reference;
      typedef typename map::const_iterator::iterator_category iterator_category;
      typedef typename map::const_iterator const_iterator;
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
    id_iterator id_begin() const { return id_iterator( map_.begin() ); }
    id_iterator id_end() const { return id_iterator( map_.end() ); }
    size_t id_size() const { return map_.size(); }
  private:
    C collection_;
    map map_;
  };
  
}

#endif
