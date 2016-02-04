#ifndef FastSimulation_Utilities_FamosDebug_H
#define FastSimulation_Utilities_FamosDebug_H

// Uncomment the following line to include the debugging code in the 
// library

// getClosestCell
//#define DEBUGGCC
// GetWindow
//#define DEBUGGW

//#define DEBUGCELLLINE

// Don't change the following
#ifdef DEBUGGCC
#define FAMOSDEBUG
#endif

#ifdef DEBUGGW
#define FAMOSDEBUG
#endif

#ifdef DEBUGCELLLINE
#define FAMOSDEBUG
#endif

#ifdef FAMOSDEBUG
#include "FastSimulation/Utilities/interface/Histos.h"
#endif

#endif

