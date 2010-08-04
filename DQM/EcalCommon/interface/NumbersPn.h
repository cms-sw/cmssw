// $Id: NumbersPn.h,v 1.72 2010/03/09 09:01:50 dellaric Exp $

/*!
  \file NumbersPn.h
  \brief Some "id" conversions
  \version $Revision: 1.72 $
  \date $Date: 2010/03/09 09:01:50 $
*/

#ifndef NUMBERSPN_H
#define NUMBERSPN_H

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
