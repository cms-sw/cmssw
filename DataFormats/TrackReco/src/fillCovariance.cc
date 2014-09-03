#include "DataFormats/TrackReco/interface/fillCovariance.h"

namespace reco
{

PerigeeCovarianceMatrix & fillCovariance(PerigeeCovarianceMatrix &v, const float *data)
{
    typedef unsigned int index;
    index idx = 0;
    for (index i = 0; i < 5; ++i) {
        for (index j = 0; j <= i; ++j) {
            v(i, j) = data[idx++];
        }
    }
    return v;
}

} // namespace reco

