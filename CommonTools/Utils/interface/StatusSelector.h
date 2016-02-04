#ifndef RecoAlgos_StatusSelector_h
#define RecoAlgos_StatusSelector_h
/* \class StatusSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StatusSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */
#include <vector>
#include <algorithm>

struct StatusSelector {
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
  } 
  template<typename T>
  bool operator()( const T & t ) const { 
    return std::find( begin_, end_, t.status() ) != end_;
  }
private:
  std::vector<int> status_;
  std::vector<int>::const_iterator begin_, end_;
};

#endif
