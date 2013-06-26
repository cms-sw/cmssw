// The code below figures out a type from the type of a pointer,
// reference, or the type itself

#ifndef GENERS_IOREFERREDTYPE_HH_
#define GENERS_IOREFERREDTYPE_HH_

#include "Alignment/Geners/interface/IOPtr.hh"
#include "Alignment/Geners/interface/StrippedType.hh"

namespace gs {
    template <class T>
    struct IOReferredType
    {
        typedef typename StrippedType<T>::type type;
    };

    // Qualifiers cannot be applied to references themselves,
    // only to the types they refer to
    template <class T>
    struct IOReferredType<T&>
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<IOPtr<T> >
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<const IOPtr<T> >
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<volatile IOPtr<T> >
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<const volatile IOPtr<T> >
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<IOProxy<T> >
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<const IOProxy<T> >
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<volatile IOProxy<T> >
    {
        typedef typename StrippedType<T>::type type;
    };

    template <class T>
    struct IOReferredType<const volatile IOProxy<T> >
    {
        typedef typename StrippedType<T>::type type;
    };
}

#endif // GENERS_IOREFERREDTYPE_HH_

