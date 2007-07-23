#ifndef RecoAlgos_StatusSelector_h
#define RecoAlgos_StatusSelector_h
/* \class StatusSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StatusSelector.h,v 1.2 2007/05/04 14:43:35 llista Exp $
 */
#include <vector>
#include <algorithm>
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

template<typename T>
struct StatusSelector {
  typedef T value_type;
  StatusSelector( const std::vector<int> & status ) { 
    for( std::vector<int>::const_iterator i = status.begin(); i != status.end(); ++ i )
      status_.push_back( * i );
    begin_ = status_.begin();
    end_ = status_.end();
  }
  StatusSelector( const StatusSelector & o ) :
    status_( o.status_ ), begin_( status_.begin() ), end_( status_.end() ) { }
  StatusSelector & operator==( const StatusSelector & o ) {
    * this = o; return * this;
  }  bool operator()( const value_type & t ) const { 
    return std::find( begin_, end_, reco::status( t ) ) != end_;
  }
private:
  std::vector<int> status_;
  std::vector<int>::const_iterator begin_, end_;
};

#endif
