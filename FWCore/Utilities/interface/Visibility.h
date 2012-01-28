#ifndef VISIBILITY_MACROS_H
#define VISIBILITY_MACROS_H
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#if GCC_PREREQUISITE(4,4,0)
#define dso_export    __attribute__ ((visibility ("default")))
#define dso_hidden    __attribute__ ((visibility ("hidden")) )
#define dso_internal  __attribute__ ((visibility ("internal")))
#define dso_protected __attribute__ ((visibility ("protected")))
#else
#define dso_export  
#define dso_hidden   
#define dso_internal 
#define dso_protected
#endif


#endif // VISIBILITY

