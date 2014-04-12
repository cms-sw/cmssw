#ifndef GENERS_IOISWRITABLE_HH_
#define GENERS_IOISWRITABLE_HH_

#include "Alignment/Geners/interface/IOIsClassType.hh"

namespace gs {
    template <typename T>
    class IOIsWritableHelper
    {
    private:
        template<bool (T::*)(std::ostream&) const> struct tester;
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(tester<&C::write>*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(IOIsWritableHelper<T>::template test<T>(0)) == 1};
    };


    template<typename T, bool is_class_type=IOIsClassType<T>::value>
    struct IOIsWritable
    {
        enum {value = 0};
    };


    template <typename T>
    struct IOIsWritable<T, true>
    {
        enum {value = IOIsWritableHelper<T>::value};
    };
}

#endif // GENERS_IOISWRITABLE_HH_

