#ifndef GENERS_IOISCLASSTYPE_HH_
#define GENERS_IOISCLASSTYPE_HH_

namespace gs {
    template <typename T>
    class IOIsClassType
    {
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(int C::*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(IOIsClassType<T>::template test<T>(0)) == 1};
    };
}

#endif // GENERS_IOISCLASSTYPE_HH_

