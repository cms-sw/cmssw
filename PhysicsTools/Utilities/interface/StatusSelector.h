#ifndef RecoAlgos_StatusSelector_h
#define RecoAlgos_StatusSelector_h
/* \class StatusSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StatusSelector.h,v 1.1 2007/02/23 15:12:41 llista Exp $
 */
#include <vector>
#include <algorithm>

template<typename T>
struct StatusSelector {
  typedef T value_type;
  StatusSelector( const std::vector<int> & status ) { 
    for( std::vector<int>::const_iterator i = status.begin(); i != status.end(); ++ i )
      status_.push_back( abs( * i ) );
     begin = status_.begin();
     end = status_.end();
  }
  bool operator()( const value_type & t ) const { 
    int status = t.status();
    return std::find( begin, end, status ) != end;
  }
private:
  std::vector<int> status_;
  std::vector<int>::const_iterator begin, end;
};

#endif
