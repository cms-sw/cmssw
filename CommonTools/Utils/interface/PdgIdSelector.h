#ifndef CommonTools_Utils_PdgIdSelector_h
#define CommonTools_Utils_PdgIdSelector_h
/* \class PdgIdSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PdgIdSelector.h,v 1.4 2007/11/14 10:14:21 llista Exp $
 */
#include <vector>
#include <algorithm>

struct PdgIdSelector {
  PdgIdSelector( const std::vector<int> & pdgId ) { 
    for( std::vector<int>::const_iterator i = pdgId.begin(); i != pdgId.end(); ++ i )
      pdgId_.push_back( abs( * i ) );
     begin_ = pdgId_.begin();
     end_ = pdgId_.end();
  }
  PdgIdSelector( const PdgIdSelector & o ) :
    pdgId_( o.pdgId_ ), begin_( pdgId_.begin() ), end_( pdgId_.end() ) { }
  PdgIdSelector & operator==( const PdgIdSelector & o ) {
    * this = o; return * this;
  }
  template<typename T>
  bool operator()( const T & t ) const { 
    return std::find( begin_, end_, abs( t.pdgId() ) ) != end_;
  }
private:
  std::vector<int> pdgId_;
  std::vector<int>::const_iterator begin_, end_;
};

#endif
