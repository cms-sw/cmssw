#ifndef Common_RangeMap_h
#define Common_RangeMap_h
#include <map>
#include <vector>
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/traits.h"

// $Id: RangeMap.h,v 1.8 2006/02/16 08:29:26 tboccali Exp $
namespace edm {
  
  template<typename ID, typename C, typename P>
    class RangeMap {
    public:
    typedef typename C::value_type value_type;
    typedef typename C::const_iterator const_iterator;
    typedef std::pair<typename C::size_type, typename C::size_type> pairType;
    typedef std::map<ID, pairType> mapType;
    typedef std::pair<const_iterator, const_iterator> range;
    
    std::vector<ID> ids() const {
      std::vector<ID> temp;
      for (typename mapType::const_iterator i = map_.begin();
	   i != map_.end();
	   i++){
	temp.push_back((*i).first);
      }
      return temp;
    }
        

    private:
    template<typename COMP>
    struct Comparator{
    public: 
      Comparator( const COMP & c ) : comp ( c ) { }
      bool operator()(const  pairType & d1,  const pairType & d2) const {
	return comp( d1.first, d2.first );
      }
      COMP comp;
    };

    public:
    template<typename COMP> range get(ID id, COMP comparator){
      //
      // use equal_range algo; expects the thing to be sorted!!!!!
      //
      pair<mapType::const_iterator, mapType::const_iterator> r = 
	equal_range(map_.begin(), map_.end(),id, Comparator<COMP>( comparator ) ); 
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
      C tempCollection;
      //     id_iterator it;
      typename mapType::iterator  it;
      for (it = map_.begin(); it != map_.end(); it ++){   
	//	      for (it = id_begin(); it != id_end(); it ++){ 
	range range_ = get((*it).first);
	typename C::size_type  begIt = tempCollection.size();
	
	copy(range_.first, range_.second, back_inserter(tempCollection));
	
	typename C::size_type endIt = tempCollection.size();
	it->second = std::make_pair( begIt, endIt );
      }
      collection_ = tempCollection;
    }
    

    

    id_iterator id_begin() const { return id_iterator( map_.begin() ); }
    id_iterator id_end() const { return id_iterator( map_.end() ); }
    size_t id_size() const { return map_.size(); }
    private:
    C collection_;
    mapType map_;
  };


  template<typename  ID, typename C, typename P > 
    struct edm::has_postinsert_trait<edm::RangeMap<ID,C,P> > 
    { 
      static bool const value = true; 
    }; 
  
}

#endif
