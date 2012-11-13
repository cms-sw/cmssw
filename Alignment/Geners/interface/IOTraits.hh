#ifndef GENERS_IOTRAITS_HH_
#define GENERS_IOTRAITS_HH_

#include <iostream>

#include "Alignment/Geners/interface/IOIsPOD.hh"
#include "Alignment/Geners/interface/IOIsPair.hh"
#include "Alignment/Geners/interface/IOIsString.hh"
#include "Alignment/Geners/interface/IOIsTuple.hh"
#include "Alignment/Geners/interface/IOIsContainer.hh"
#include "Alignment/Geners/interface/IOIsWritable.hh"
#include "Alignment/Geners/interface/IOIsReadable.hh"
#include "Alignment/Geners/interface/IOIsExternal.hh"
#include "Alignment/Geners/interface/IOIsSharedPtr.hh"
#include "Alignment/Geners/interface/IOIsIOPtr.hh"
#include "Alignment/Geners/interface/IOIsClassType.hh"
#include "Alignment/Geners/interface/IOIsContiguous.hh"

namespace gs {
    template <class T>
    struct IOTraits
    {
        static const bool IsClass = IOIsClassType<T>::value;
        enum {ISCLASS = 1};

        // Pointers are not PODs for I/O purposes.
        //
        // It looks like std::array of PODs is itself considered
        // a POD by the "CPP11_is_pod" template, at least by some
        // g++ versions (in particular, 4.4.5). This is a bug/feature
        // that we need to avoid.
        static const bool IsPOD = (IOIsPOD<T>::value &&
                                   !IOIsContainer<T>::value &&
                                   !CPP11_is_pointer<T>::value &&
                                   !IOIsExternal<T>::value);
        enum {ISPOD = 2};

        static const bool IsWritable = (IOIsWritable<T>::value &&
                                        !IOIsExternal<T>::value);
        enum {ISWRITABLE = 4};

        static const bool IsStdContainer = (IOIsContainer<T>::value && 
                                            !IOIsWritable<T>::value &&
                                            !IOIsExternal<T>::value);
        enum {ISSTDCONTAINER = 8};

        // Readable objects are required to be writable
        static const bool IsPlaceReadable = (IOIsPlaceReadable<T>::value &&
                                             IOIsWritable<T>::value);
        enum {ISPLACEREADABLE = 16};

        // Prefer place readability to heap readability
        static const bool IsHeapReadable = (IOIsHeapReadable<T>::value &&
                                            !IOIsPlaceReadable<T>::value &&
                                            IOIsWritable<T>::value);
        enum {ISHEAPREADABLE = 32};

        static const bool IsPointer = (CPP11_is_pointer<T>::value &&
                                       !IOIsExternal<T>::value);
        enum {ISPOINTER = 64};

        static const bool IsSharedPtr = (IOIsSharedPtr<T>::value &&
                                         !IOIsExternal<T>::value);
        enum {ISSHAREDPTR = 128};

        static const bool IsPair = IOIsPair<T>::value;
        enum {ISPAIR = 256};

        static const bool IsString = IOIsString<T>::value;
        enum {ISSTRING = 512};

        // The following trait is relevant for containers only
        static const bool IsContiguous = IOIsContiguous<T>::value;
        enum {ISCONTIGUOUS = 1024};

        static const bool IsTuple = IOIsTuple<T>::value;
        enum {ISTUPLE = 2048};

        static const bool IsIOPtr = IOIsIOPtr<T>::value;
        enum {ISIOPTR = 4096};

        // A catch-all definition for externally defined types which
        // want to use the template-based I/O within this system but
        // do not want to implement the standard "read/write" mechanism.
        // The user has to declare the external type by modifying the
        // "IOIsExternal" template.
        static const bool IsExternal = IOIsExternal<T>::value;
        enum {ISEXTERNAL = 8192};

        // Special enums for heap-readable objects known
        // to be called with zero pointer as an argument.
        // In this case we will avoid compiling an assignment
        // operator for the object.
        enum {ISNULLPOINTER = 16384};
        enum {ISPUREHEAPREADABLE = 32768};

        static const int Signature = 
            IsClass*ISCLASS + 
            IsPOD*ISPOD +
            IsWritable*ISWRITABLE +
            IsStdContainer*ISSTDCONTAINER + 
            IsPlaceReadable*ISPLACEREADABLE +
            IsHeapReadable*ISHEAPREADABLE + 
            IsPointer*ISPOINTER + 
            IsSharedPtr*ISSHAREDPTR + 
            IsPair*ISPAIR +
            IsString*ISSTRING + 
            IsContiguous*ISCONTIGUOUS + 
            IsTuple*ISTUPLE +
            IsIOPtr*ISIOPTR +
            IsExternal*ISEXTERNAL +
            IsPointer*ISNULLPOINTER +
            IsHeapReadable*ISPUREHEAPREADABLE;
    };

    template <class T>
    inline IOTraits<T> IOItemTraits(const T&) {return IOTraits<T>();}
}

template <class T>
std::ostream& operator<<(std::ostream& os, const gs::IOTraits<T>&)
{
    typedef gs::IOTraits<T> Tr;
    os << "IsClass = " << Tr::IsClass
       << ", IsPOD = " << Tr::IsPOD
       << ", IsWritable = " << Tr::IsWritable
       << ", IsStdContainer = " << Tr::IsStdContainer
       << ", IsPlaceReadable = " << Tr::IsPlaceReadable
       << ", IsHeapReadable = " << Tr::IsHeapReadable
       << ", IsPointer = " << Tr::IsPointer
       << ", IsSharedPtr = " << Tr::IsSharedPtr
       << ", IsPair = " << Tr::IsPair
       << ", IsString = " << Tr::IsString
       << ", IsContiguous = " << Tr::IsContiguous
       << ", IsTuple = " << Tr::IsTuple
       << ", IsIOPtr = " << Tr::IsIOPtr
       << ", IsExternal = " << Tr::IsExternal;
    return os;
}

#endif // GENERS_IOTRAITS_HH_

