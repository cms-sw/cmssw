//-------------------------------------------------
//
//   Description: Parameters for Assignment
//
//
//   $Date: 2007/02/27 11:44:00 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------
#ifndef L1MUBM_ASS_PARAM_H
#define L1MUBM_ASS_PARAM_H

#include <iosfwd>

// maximal number of pt assignment methods
const int MAX_PTASSMETH = 13;

// pt assignment methods
enum PtAssMethod { PT12L,  PT12H,  PT13L,  PT13H,  PT14L,  PT14H,
                   PT23L,  PT23H,  PT24L,  PT24H,  PT34L,  PT34H, 
                   NODEF };


// overload output stream operator for pt-assignment methods
std::ostream& operator<<(std::ostream& s, PtAssMethod method);

#endif
