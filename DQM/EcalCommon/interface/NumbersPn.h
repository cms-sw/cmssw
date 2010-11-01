#ifndef NumbersPn_H
#define NumbersPn_H

/*!
  \file NumbersPn.h
  \brief Some "id" conversions
  \version $Revision: 1.8 $
  \date $Date: 2010/08/06 12:28:07 $
*/

#include <string>
#include <stdexcept>
#include <vector>

class NumbersPn {

 public:

  static int ipnEE( const int ism, const int ipnid ) throw( std::runtime_error );

  static void getPNs( const int ism, const int ix, const int iy, std::vector<int>& PNsInLM ) throw( std::runtime_error );

  static int iLM( const int ism, const int ix, const int iy ) throw( std::runtime_error );

};

#endif // NumbersPn_H
