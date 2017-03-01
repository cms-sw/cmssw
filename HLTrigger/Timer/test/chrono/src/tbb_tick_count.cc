#ifdef HAVE_TBB

#include "interface/tbb_tick_count.h"

const tbb::tick_count clock_tbb_tick_count::epoch = tbb::tick_count::now();

#endif // HAVE_TBB
