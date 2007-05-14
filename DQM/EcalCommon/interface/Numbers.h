// $Id: Numbers.h,v 1.2 2007/05/08 12:07:19 benigno Exp $

/*!
  \file Numbers.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.2 $
  \date $Date: 2007/05/08 12:07:19 $
*/

#ifndef Numbers_H
#define Numbers_H

#include <string>
#include <stdexcept>

class Numbers {

 public:

  static int         iEB( int ism ) throw( std::runtime_error );

  static std::string sEB( int ism ) throw( std::runtime_error );

  static int         iEE( int ism ) throw( std::runtime_error );

  static std::string sEE( int ism ) throw( std::runtime_error );

};

#endif // Numbers_H
