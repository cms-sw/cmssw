#ifndef OnlineDBOracle_Oracle_h
#define OnlineDBOracle_Oracle_h

// Isolate the dependency on Oracle to this header.
#include "occi.h"  //INCLUDECHECKER:SKIP
// Unddefine any dangerous defines in occi.h.
#ifdef CONST
#undef CONST
#endif

#endif
