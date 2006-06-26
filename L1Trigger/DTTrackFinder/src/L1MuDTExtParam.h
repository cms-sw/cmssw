//-------------------------------------------------
//
//   Description: Parameters for Extrapolation
//
//
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------
#ifndef L1MUDT_EXT_PARAM_H
#define L1MUDT_EXT_PARAM_H

#include <iosfwd>

//max. number of Extrapolations
const int MAX_EXT = 12;

// extrapolation types
enum Extrapolation { EX12, EX13, EX14, EX21, EX23, EX24, EX34,
                     EX15, EX16, EX25, EX26, EX56 };


// overload output stream operator for Extrapolation
ostream& operator<<( ostream& s, Extrapolation ext);

#endif
