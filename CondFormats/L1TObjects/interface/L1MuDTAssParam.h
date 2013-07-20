//-------------------------------------------------
//
//   Description: Parameters for Assignment
//
//
//   $Date: 2007/03/30 07:48:02 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------
#ifndef L1MUDT_ASS_PARAM_H
#define L1MUDT_ASS_PARAM_H

#include <iosfwd>

// maximal number of pt assignment methods
const int MAX_PTASSMETH = 28;

// pt assignment methods
enum PtAssMethod { PT12L,  PT12H,  PT13L,  PT13H,  PT14L,  PT14H,
                   PT23L,  PT23H,  PT24L,  PT24H,  PT34L,  PT34H, 
                   PT12LO, PT12HO, PT13LO, PT13HO, PT14LO, PT14HO,
                   PT23LO, PT23HO, PT24LO, PT24HO, PT34LO, PT34HO, 
                   PT15LO, PT15HO, PT25LO, PT25HO, NODEF };


// overload output stream operator for pt-assignment methods
std::ostream& operator<<(std::ostream& s, PtAssMethod method);

#endif
