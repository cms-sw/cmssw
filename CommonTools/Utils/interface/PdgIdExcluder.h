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
  explicit PdgIdExcluder( const std::vector<int> & pdgId ) {
    pdgId_.reserve(pdgId.size());
    for( int i : pdgId) {
      pdgId_.push_back( abs( i ) );
    }
  }
  template<typename T>
  bool operator()( const T & t ) const { 
    return std::find( pdgId_.begin(), pdgId_.end(), abs( t.pdgId() ) ) == pdgId_.end();
  }
  private:
  std::vector<int> pdgId_;
};

#endif
