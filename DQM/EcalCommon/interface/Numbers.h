// $Id: $

/*!
  \file Numbers.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: $
  \date $Date: $
*/

#ifndef Numbers_H
#define Numbers_H

#include <string>
#include <stdexcept>

class Numbers {

 public:

  static int         iEB( int fed ) throw( std::runtime_error );

  static std::string sEB( int fed ) throw( std::runtime_error );

};

#endif // Numbers_H
