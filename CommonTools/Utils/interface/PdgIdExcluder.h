#ifndef CommonTools_Utils_PdgIdExcluder_h
#define CommonTools_Utils_PdgIdExcluder_h
/* \class PdgIdExcluder
 *
 * \author Luca Lista, INFN
 *
 * $Id: PdgIdExcluder.h,v 1.1 2009/02/24 15:01:17 llista Exp $
 */
#include <vector>
#include <algorithm>

struct PdgIdExcluder {
  PdgIdExcluder( const std::vector<int> & pdgId ) { 
    for( std::vector<int>::const_iterator i = pdgId.begin(); i != pdgId.end(); ++ i )
      pdgId_.push_back( abs( * i ) );
    begin_ = pdgId_.begin();
    end_ = pdgId_.end();
  }
  PdgIdExcluder( const PdgIdExcluder & o ) :
    pdgId_( o.pdgId_ ), begin_( pdgId_.begin() ), end_( pdgId_.end() ) { }
  PdgIdExcluder & operator=( const PdgIdExcluder & o ) {
    * this = o; return * this;
  }
  template<typename T>
  bool operator()( const T & t ) const { 
    return std::find( begin_, end_, abs( t.pdgId() ) ) == end_;
  }
  private:
  std::vector<int> pdgId_;
  std::vector<int>::const_iterator begin_, end_;
};

#endif
