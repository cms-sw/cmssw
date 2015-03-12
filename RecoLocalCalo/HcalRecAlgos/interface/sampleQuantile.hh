#ifndef SAMPLEQUANTILE_HH_
#define SAMPLEQUANTILE_HH_

#include <algorithm>

template<unsigned N, typename Real>
Real sampleQuantile(const Real* data, const double q)
{
    Real tmp[N];
    {
        Real *to = &tmp[0];
        for (unsigned i=0; i<N; ++i)
            *to++ = *data++;
    }
    std::sort(tmp, tmp+N);

    // Map q = 0.0 to tmp[0] and q = 1.0 to tmp[N-1]
    const unsigned nm1 = N-1U;
    if (q <= 0.0)
        return tmp[0];
    else if (q >= 1.0)
        return tmp[nm1];
    else
    {
        const double n = q*nm1;
        const unsigned un = n;
        const double w = n - un;
        return (1.0 - w)*tmp[un] + w*tmp[un+1U];
    }
}

#endif // SAMPLEQUANTILE_HH_
