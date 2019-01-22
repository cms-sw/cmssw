#ifndef PolyType_h
#define PolyType_h

#include <boost/iterator/iterator_facade.hpp>
#include <boost/operators.hpp>
#include <iostream>
#include <list>
#include <set>

template<class T> class poly;
template<class T> poly<T> operator+ (const poly<T>&, const char* ); 
template<class T> poly<T> operator+ (const char*, const poly<T>& ); 

template<class T>
class poly :
  boost::incrementable< poly<T>,
  boost::addable< poly<T>,
  boost::multipliable< poly<T>,
  boost::multipliable2< poly<T>, T,
  boost::less_than_comparable< poly<T> > > > > > {    
    
    std::list<std::set<T> > columns;
    
  public:

    class const_iterator;
    typedef T value_type;
    typedef typename std::list<std::set<T> >::iterator                column_iterator;
    typedef typename std::list<std::set<T> >::const_iterator    const_column_iterator;
    poly() {}
    poly(const T& t) {operator+=(t);}

    bool operator<(const poly& R) const { 
      const_column_iterator column(columns.begin()), Rcolumn(R.columns.begin());
      while( column!=columns.end() && Rcolumn!=R.columns.end() && *column==*Rcolumn) { ++column; ++Rcolumn; }
      return column!=columns.end() && Rcolumn!=R.columns.end() && *column < *Rcolumn;
    }  
    poly operator++() {columns.push_back(std::set<T>()); return *this;}                    
    poly operator+=(const poly& R) { columns.insert(columns.end(),R.columns.begin(),R.columns.end()); return *this;}
    poly operator+=(const T& r) { operator++(); return operator*=(r);}
    poly operator*=(const T& r) { columns.back().insert(r); return *this;}    
    poly operator*=(const poly& R) { columns.back().insert(R.begin(),R.end()); return *this;}    
    friend poly<T> operator+ <> (const poly<T>&, const char*);
    friend poly<T> operator+ <> (const char*, const poly<T>&);

    const_iterator begin() const { return const_iterator(*this);}
    const_iterator end()   const { return const_iterator::end_of(*this);} 

    auto const& getColumns() const { return columns; }
    auto& getColumns() { return columns; }

    size_t size() const { 
      if(columns.empty()) return 0;
      size_t size=1;
      for( const_column_iterator column = columns.begin(); column != columns.end(); ++column) 
	size *= column->size(); 
      return size;
    }

    class const_iterator 
      : public boost::iterator_facade< const_iterator, T const, boost::bidirectional_traversal_tag, T >  {
      friend class boost::iterator_core_access;

      std::list<typename std::set<T>::const_iterator>    state;
      typename std::list<std::set<T> >::const_iterator   begin, end;

      typedef typename std::list<typename std::set<T>::const_iterator>::iterator                state_iterator;
      typedef typename std::list<typename std::set<T>::const_iterator>::const_iterator    const_state_iterator;

      bool equal(const_iterator const& rhs) const { return std::equal( state.begin(), state.end(), rhs.state.begin() ); }
      T dereference() const { T s; for(const_state_iterator istate=state.begin(); istate!=state.end(); ++istate)  s+= **istate; return s; }
      void increment() { 
	state_iterator istate = state.begin();
	const_column_iterator column = begin;
	while( column != end && ++*istate == column->end() ) { ++istate; ++column;} 
	if( column == end ) {--column; --istate;}
	while( istate != state.begin() ) {--istate; *istate = (--column)->begin();}
      }
      void decrement() {  
	state_iterator istate = state.begin();
	const_column_iterator column = begin;
	while( column != end && *istate == column->begin())  { ++istate; ++column;}
	if( column != end) --*istate;
	while( istate != state.begin() ) {--istate; *istate = --((--column)->end());} 
      }
      
    public:
      
      const_iterator() {}
      const_iterator(const poly& p) : begin(p.getColumns().begin()), end(p.getColumns().end()) {
	const_column_iterator column = begin; 
	while(column!=end) state.push_back((column++)->begin()); 
      }
      static const_iterator end_of(const poly& p) {
	const_iterator it(p);
	if(p.size()!=0) *--(it.state.end()) = (--p.getColumns().end())->end();
	return it;
      }
      
  };    

};

template<class T> poly<T> operator+ (const poly<T>& lhs, const char* rhs ) { return lhs + poly<T>(rhs);} 
template<class T> poly<T> operator+ (const char* lhs, const poly<T>& rhs ) { return poly<T>(lhs) + rhs;}

template <class charT, class traits, class T> 
inline
std::basic_ostream<charT,traits>& operator<<(std::basic_ostream<charT,traits>& strm, const poly<T>& f) { 
  for(auto const& column : f.getColumns()) 
    { strm << "( "; for(auto const& entry : column) strm << entry << ", "; strm << " )" << std::endl; }
  return strm; 
}

#endif
