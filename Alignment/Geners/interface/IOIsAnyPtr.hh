#ifndef GENERS_IOISANYPTR_HH_
#define GENERS_IOISANYPTR_HH_

#include "Alignment/Geners/interface/CPP11_type_traits.hh"
#include "Alignment/Geners/interface/IOIsSharedPtr.hh"
#include "Alignment/Geners/interface/IOIsIOPtr.hh"

namespace gs {
    // Here we are taking into account only the types of pointers
    // known to the I/O system
    template <class T>
    struct IOIsAnyPtr
    {
        static const bool value = (CPP11_is_pointer<T>::value ||
                                   IOIsSharedPtr<T>::value ||
                                   IOIsIOPtr<T>::value);
    };
}

#endif // GENERS_IOISANYPTR_HH_

