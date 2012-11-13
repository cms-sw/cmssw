//=========================================================================
// streamposIO.hh
//
// Specialize read_pod and write_pod so that they also
// work with std::streampos.
//
// The code for storing the stream offset is necessarily
// going to be implementation-dependent. The C++ standard
// does not specify std::streampos with enough detail.
//
// I. Volobouev
// March 2011
//=========================================================================

#ifndef GENERS_STREAMPOSIO_HH_
#define GENERS_STREAMPOSIO_HH_

#include "Alignment/Geners/interface/binaryIO.hh"

namespace gs {
    template<>
    inline void write_pod<std::streampos>(std::ostream& of,
                                          const std::streampos& s)
    {
        std::streamoff off(s);
        long long loc = off;
        write_pod(of, loc);
    }

    template<>
    inline void read_pod<std::streampos>(std::istream& in,
                                         std::streampos* ps)
    {
        assert(ps);
        long long loc = 0LL;
        read_pod(in, &loc);
        std::streamoff off(loc);
        *ps = std::streampos(off);
    }

    template<>
    inline void write_pod_array<std::streampos>(std::ostream& of,
                                                const std::streampos* pod,
                                                const unsigned long len)
    {
        if (len)
        {
            assert(pod);
            for (unsigned long i=0; i<len; ++i)
                write_pod(of, pod[i]);
        }
    }

    template<>
    inline void read_pod_array<std::streampos>(std::istream& in,
                                               std::streampos* pod,
                                               const unsigned long len)
    {
        if (len)
        {
            assert(pod);
            for (unsigned long i=0; i<len; ++i)
                read_pod(in, pod + i);
        }
    }
}

#endif // GENERS_STREAMPOSIO_HH_

