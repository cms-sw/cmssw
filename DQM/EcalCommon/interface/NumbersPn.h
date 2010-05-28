#ifndef NumbersPn_H
#define NumbersPn_H

#include <string>
#include <stdexcept>
#include <vector>

class NumbersPn {

 public:

  static int ipnEE( const int ism, const int ipnid ) throw( std::runtime_error );

  static int getPN( const int ism, const int ix, const int iy ) throw( std::runtime_error );

  static std::vector<int> getPNs( const int ism, const int ix, const int iy ) throw( std::runtime_error );

  static int iLM( const int ism, const int ix, const int iy ) throw( std::runtime_error );

};

#endif // NumbersPn_H
