
#ifndef __Rsc_global_func__
#define __Rsc_global_func__

#include "TRandom3.h"
#include "TH1.h"

namespace Rsc
    {
    extern TRandom3 random_generator;

    /// Get the center of the histo
    double getHistoCenter(TH1* histo, double n_rms=1,bool display_result=false);

    /// Get the "effective sigmas" of the histo
    double* getHistoPvals (TH1F* histo, double percentage);

    /// Get the median of an histogram
    double getMedian(TH1* histo);

    }
#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
