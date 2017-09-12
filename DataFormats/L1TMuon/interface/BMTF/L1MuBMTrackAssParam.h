//-------------------------------------------------
//
//   Description: Parameters for Track Assembler
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------
#ifndef L1MUBM_TRACK_ASS_PARAM_H
#define L1MUBM_TRACK_ASS_PARAM_H

#include <iosfwd>
#include <string>

//@@ number of Track Classes
const int MAX_TC = 11;

//@@ defined Track Classes ordered in decreasing priority
enum TrackClass { T1234, T123, T124, T134, T234,
                  T12,   T14,  T13,  T24,  T23,  T34, UNDEF };

// overload output stream operator for TrackClass
std::ostream& operator<<( std::ostream& s, TrackClass tc);

// convert TrackClass to bitmap
const unsigned int tc2bitmap(const TrackClass tc);

// convert TrackClass to graphical format
const std::string tc2string(const TrackClass tc);

#endif
