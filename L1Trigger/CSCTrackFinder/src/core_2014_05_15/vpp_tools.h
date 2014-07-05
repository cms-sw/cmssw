#include <iostream>
#include <fstream>

// concatenation of two signal_s
#define sigcat(s1,n1,s2,n2) ((s1 << n2) | s2)

// signal_ declaration
#define Signal(s) ULLONG s, RC_##s, CH_##s

// signal_ initialization
#define siginit(s) {if ((CH_##s = s - RC_##s) != 0) {__glob__change__ = 1; RC_##s = s;}}

// memory initialization
#define meminit(s) for (unsigned __mem_ind__ = 0; __mem_ind__ < sizeof(s)/sizeof(ULLONG); __mem_ind__++) siginit(s[__mem_ind__]);

