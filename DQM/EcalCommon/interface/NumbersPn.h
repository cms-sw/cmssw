#ifndef NUMBERSPN_H
#define NUMBERSPN_H

/*!
  \file NumbersPn.h
  \brief Some "id" conversions
  \version $Revision: 1.10 $
  \date $Date: 2010/08/20 19:09:04 $
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

#endif // NUMBERSPN_H
