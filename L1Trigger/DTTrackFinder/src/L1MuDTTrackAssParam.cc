//-------------------------------------------------
//
//   Description: Parameters for Track Assembler
//
//
//   $Date: 2007/02/27 11:44:00 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackAssParam.h"
#include <iostream>

using namespace std;

//
// output stream operator for TrackClass
//
ostream& operator<<( ostream& s, TrackClass tc) {

  switch ( tc ) {
    case T1234: return s << "T1234 ";
    case T123:  return s << "T123  ";
    case T124:  return s << "T124  ";
    case T134:  return s << "T134  ";
    case T234:  return s << "T234  ";
    case T12:   return s << "T12   ";
    case T14:   return s << "T14   ";
    case T13:   return s << "T13   ";
    case T24:   return s << "T24   ";
    case T23:   return s << "T23   ";
    case T34:   return s << "T34   ";
    default: return s << "UNDEF ";
  }

}


//
// convert TrackClass to bitmap
//
const unsigned int tc2bitmap(const TrackClass tc) {

  unsigned int value = 0;

  switch ( tc ) {
    case T1234: { value = 15; break; }
    case T123:  { value =  7; break; }
    case T124:  { value = 11; break; } 
    case T134:  { value = 13; break; }
    case T234:  { value = 14; break; }
    case T12:   { value =  3; break; }
    case T14:   { value =  9; break; }
    case T13:   { value =  5; break; }
    case T24:   { value = 10; break; }
    case T23:   { value =  6; break; } 
    case T34:   { value = 12; break; }
    default:    { value =  0; break; }
  }

  return value;

}


//
// convert TrackClass graphical format
//
const string tc2string(const TrackClass tc) {

  string str = "####";

  switch ( tc ) {
    case T1234: { str = "****"; break; }
    case T123:  { str = "***-"; break; }
    case T124:  { str = "**-*"; break; } 
    case T134:  { str = "*-**"; break; }
    case T234:  { str = "-***"; break; }
    case T12:   { str = "**--"; break; }
    case T14:   { str = "*--*"; break; }
    case T13:   { str = "*-*-"; break; }
    case T24:   { str = "-*-*"; break; }
    case T23:   { str = "-**-"; break; } 
    case T34:   { str = "--**"; break; }
    default:    { str = "UNDEF"; break; }
  }

  return str;

}
