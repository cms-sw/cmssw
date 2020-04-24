#ifndef GENERS_REFERENCE_HH_
#define GENERS_REFERENCE_HH_

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"
#include "Alignment/Geners/interface/AbsReference.hh"

namespace gs {
    template <typename T>
    class Reference : public AbsReference
    {
    public:
        inline Reference(AbsArchive& ar, const unsigned long long itemId)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::Single", itemId) {}

        inline Reference(AbsArchive& ar, const char* name, const char* category)
            :  AbsReference(ar, ClassId::makeId<T>(), "gs::Single",
                            name, category) {}

#ifndef SWIG
        inline Reference(AbsArchive& ar, const std::string& name,
                         const char* category)
            :  AbsReference(ar, ClassId::makeId<T>(), "gs::Single",
                            name.c_str(), category) {}

        inline Reference(AbsArchive& ar, const char* name,
                         const std::string& category)
            :  AbsReference(ar, ClassId::makeId<T>(), "gs::Single",
                            name, category.c_str()) {}

        inline Reference(AbsArchive& ar, const std::string& name,
                         const std::string& category)
            :  AbsReference(ar, ClassId::makeId<T>(), "gs::Single",
                            name.c_str(), category.c_str()) {}
#endif

        inline Reference(AbsArchive& ar, const SearchSpecifier& namePattern,
                         const SearchSpecifier& categoryPattern)
            :  AbsReference(ar, ClassId::makeId<T>(), "gs::Single",
                            namePattern, categoryPattern) {}

        // Methods to retrieve the item
        void restore(unsigned long index, T* obj) const;
        CPP11_auto_ptr<T> get(unsigned long index) const;
        CPP11_shared_ptr<T> getShared(unsigned long index) const;

    private:
        Reference();
        T* getPtr(unsigned long index) const;
    };
}

#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    template <typename T>
    inline void Reference<T>::restore(const unsigned long index, T* obj) const
    {
        const unsigned long long itemId = this->id(index);
        assert(itemId);
        restore_item(this->positionInputStream(itemId), obj, true);
    }

    template <typename T>
    inline T* Reference<T>::getPtr(const unsigned long index) const
    {
        const unsigned long long itemId = this->id(index);
        assert(itemId);
        T* barePtr = 0;
        std::vector<ClassId> state;
        if (GenericReader<std::istream, std::vector<ClassId>, T*,
            Int2Type<IOTraits<int>::ISNULLPOINTER> >::process(
                barePtr, this->positionInputStream(itemId), &state, true))
            assert(barePtr);
        else
        {
            delete barePtr;
            barePtr = 0;
        }
        if (!barePtr)
            throw IOInvalidData("In gs::Reference::getPtr: "
                                "failed to read in the object");
        return barePtr;
    }

    template <typename T>
    inline CPP11_auto_ptr<T> Reference<T>::get(const unsigned long index) const
    {
        return CPP11_auto_ptr<T>(getPtr(index));
    }

    template <typename T>
    inline CPP11_shared_ptr<T> Reference<T>::getShared(
        const unsigned long index) const
    {
        return CPP11_shared_ptr<T>(getPtr(index));
    }
}


#endif // GENERS_REFERENCE_HH_

