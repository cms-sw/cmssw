#ifndef GENERS_IOISREADABLE_HH_
#define GENERS_IOISREADABLE_HH_

#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/IOIsClassType.hh"
#include "Alignment/Geners/interface/StrippedType.hh"

namespace gs {
    template <typename T>
    class IOIsHeapReadableHelper
    {
    private:
        template<T* (*)(const ClassId&, std::istream&)> struct tester;
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(tester<&C::read>*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(IOIsHeapReadableHelper<T>::template test<T>(0)) == 1};
    };


    template<typename T, bool is_class_type=IOIsClassType<T>::value>
    struct IOIsHeapReadable
    {
        enum {value = 0};
    };


    template <typename T>
    struct IOIsHeapReadable<T, true>
    {
        enum {value = IOIsHeapReadableHelper<typename StrippedType<T>::type>::value};
    };


    template <typename T>
    class IOIsPlaceReadableHelper
    {
    private:
        template<void (*)(const ClassId&, std::istream&, T*)> struct tester;
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(tester<&C::restore>*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(IOIsPlaceReadableHelper<T>::template test<T>(0)) == 1};
    };


    template<typename T, bool is_class_type=IOIsClassType<T>::value>
    struct IOIsPlaceReadable
    {
        enum {value = 0};
    };


    template <typename T>
    struct IOIsPlaceReadable<T, true>
    {
        enum {value = IOIsPlaceReadableHelper<typename StrippedType<T>::type>::value};
    };
}

#endif // GENERS_IOISREADABLE_HH_

