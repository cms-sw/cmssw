#ifndef VISIBILITY_MACROS_H
#define VISIBILITY_MACROS_H

#define dso_export    __attribute__ ((visibility ("default")))
#define dso_hidden    __attribute__ ((visibility ("hidden")) )
#define dso_internal  __attribute__ ((visibility ("internal")))
#define dso_protected __attribute__ ((visibility ("protected")))


#endif // VISIBILITY

