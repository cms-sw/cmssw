#ifndef GENERS_ARRAYREFERENCE_HH_
#define GENERS_ARRAYREFERENCE_HH_

#include <cstddef>
#include "Alignment/Geners/interface/AbsReference.hh"

namespace gs {
    template <typename T>
    class ArrayReference : public AbsReference
    {
    public:
        inline ArrayReference(AbsArchive& ar, const unsigned long long itemId)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::Array", itemId) {}

        inline ArrayReference(AbsArchive& ar,
                              const char* name, const char* category)
            :  AbsReference(ar, ClassId::makeId<T>(), "gs::Array",
                            name, category) {}

        inline ArrayReference(AbsArchive& ar,
                              const SearchSpecifier& namePattern,
                              const SearchSpecifier& categoryPattern)
            :  AbsReference(ar, ClassId::makeId<T>(), "gs::Array",
                            namePattern, categoryPattern) {}

        // There is only one method to retrieve an array
        void restore(unsigned long idx, T* arr, std::size_t len) const;

    private:
        ArrayReference();
    };
}

#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    template <typename T>
    inline void ArrayReference<T>::restore(const unsigned long index, T* arr,
                                           const std::size_t len) const
    {
        const unsigned long long itemId = this->id(index);
        assert(itemId);
        read_array(this->positionInputStream(itemId), arr, len);
    }
}


#endif // GENERS_ARRAYREFERENCE_HH_

