#ifndef NUMBERSPN_H
#define NUMBERSPN_H

/*!
  \file NumbersPn.h
  \brief Some "id" conversions
  \version $Revision: 1.7 $
  \date $Date: 2010/08/04 06:48:52 $
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

#endif // NUMBERSPN_H
