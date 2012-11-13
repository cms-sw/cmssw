#ifndef GENERS_COMPLEXIO_HH_
#define GENERS_COMPLEXIO_HH_

#include <complex>

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOIsPOD.hh"
#include "Alignment/Geners/interface/IOIsContainer.hh"

#define gs_specialize_complex_read_write(T) /**/                       \
namespace gs {                                                         \
    template <>                                                        \
    inline void write_pod<std::complex<T> >(std::ostream& of,          \
                                            const std::complex<T>& s)  \
    {                                                                  \
        write_pod(of, s.real());                                       \
        write_pod(of, s.imag());                                       \
    }                                                                  \
    template <>                                                        \
    inline void read_pod<std::complex<T> >(std::istream& in,           \
                                           std::complex<T>* ps)        \
    {                                                                  \
        assert(ps);                                                    \
        T re, im;                                                      \
        read_pod(in, &re);                                             \
        read_pod(in, &im);                                             \
        *ps = std::complex<T>(re, im);                                 \
    }                                                                  \
    template <>                                                        \
    inline void write_pod_array<std::complex<T> >(                     \
        std::ostream& of, const std::complex<T>* pod,                  \
        const unsigned long len)                                       \
    {                                                                  \
        if (len)                                                       \
        {                                                              \
            assert(pod);                                               \
            for (unsigned long i=0; i<len; ++i)                        \
                write_pod<std::complex<T> >(of, pod[i]);               \
        }                                                              \
    }                                                                  \
    template <>                                                        \
    inline void read_pod_array<std::complex<T> >(                      \
        std::istream& in, std::complex<T>* pod,                        \
        const unsigned long len)                                       \
    {                                                                  \
        if (len)                                                       \
        {                                                              \
            assert(pod);                                               \
            T re, im;                                                  \
            for (unsigned long i=0; i<len; ++i)                        \
            {                                                          \
                read_pod(in, &re);                                     \
                read_pod(in, &im);                                     \
                pod[i] = std::complex<T>(re, im);                      \
            }                                                          \
        }                                                              \
    }                                                                  \
}

gs_specialize_complex_read_write(float)
gs_specialize_complex_read_write(double)
gs_specialize_complex_read_write(long double)

namespace gs {
    template <typename T>
    struct IOIsPOD<std::complex<T> >
    {
        enum {value = 1};
    };

    template <typename T>
    struct IOIsPOD<const std::complex<T> >
    {
        enum {value = 1};
    };

    template <typename T>
    struct IOIsPOD<volatile std::complex<T> >
    {
        enum {value = 1};
    };

    template <typename T>
    struct IOIsPOD<const volatile std::complex<T> >
    {
        enum {value = 1};
    };

    template <typename T>
    class IOIsContainer<std::complex<T> >
    {
    public:
        enum {value = 0};
    };

    template <typename T>
    class IOIsContainer<const std::complex<T> >
    {
    public:
        enum {value = 0};
    };

    template <typename T>
    class IOIsContainer<volatile std::complex<T> >
    {
    public:
        enum {value = 0};
    };

    template <typename T>
    class IOIsContainer<const volatile std::complex<T> >
    {
    public:
        enum {value = 0};
    };
}

gs_specialize_template_id_T(std::complex, 0, 1)

#endif // GENERS_COMPLEXIO_HH_

