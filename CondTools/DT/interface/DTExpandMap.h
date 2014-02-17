#ifndef DTExpandMap_H
#define DTExpandMap_H
/** \class DTExpandMap
 *
 *  Description:
 *       Class to build full readout map from compact map
 *
 *  $Date: 2009/03/26 14:11:03 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//---------------
// C++ Headers --
//---------------
#include <iostream>

class DTExpandMap {
 public:
  static void expandSteering( std::ifstream& file );
};
#endif

