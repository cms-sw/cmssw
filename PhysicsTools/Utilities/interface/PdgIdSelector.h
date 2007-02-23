#ifndef RecoAlgos_PdgIdSelector_h
#define RecoAlgos_PdgIdSelector_h
/* \class PdgIdSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PdgIdSelector.h,v 1.2 2007/01/31 14:42:59 llista Exp $
 */
#include <vector>
#include <algorithm>

template<typename T>
struct PdgIdSelector {
  typedef T value_type;
  PdgIdSelector( const std::vector<int> & pdgId ) { 
    for( std::vector<int>::const_iterator i = pdgId.begin(); i != pdgId.end(); ++ i )
      pdgId_.push_back( abs( * i ) );
     begin = pdgId_.begin();
     end = pdgId_.end();
  }
  bool operator()( const value_type & t ) const { 
    int id = abs( t.pdgId() );
    return std::find( begin, end, id ) != end;
  }
private:
  std::vector<int> pdgId_;
  std::vector<int>::const_iterator begin, end;
};

#endif
