#ifndef NumbersPn_H
#define NumbersPn_H

/*!
  \file NumbersPn.h
  \brief Some "id" conversions
  \version $Revision: 1.9 $
  \date $Date: 2010/08/08 08:16:15 $
*/

#include <string>
#include <stdexcept>
#include <vector>

class NumbersPn {

 public:

  static int ipnEE( const int ism, const int ipnid ) throw( std::runtime_error );

  static void getPNs( const int ism, const int ix, const int iy, std::vector<int>& PNsInLM ) throw( std::runtime_error );

  static int iLM( const int ism, const int ix, const int iy ) throw( std::runtime_error );

 private:

  NumbersPn() {}; // Hidden to force static use
  ~NumbersPn() {}; // Hidden to force static use

};

#endif // NumbersPn_H
