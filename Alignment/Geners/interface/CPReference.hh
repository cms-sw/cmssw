#ifndef GENERS_CPREFERENCE_HH_
#define GENERS_CPREFERENCE_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <memory>

#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/AbsReference.hh"
#include "Alignment/Geners/interface/collectTupleNames.hh"

namespace gs {
    template <class T>
    class CPReference : public AbsReference
    {
    public:
        inline CPReference(const std::vector<std::string>& columnNames,
                           AbsArchive& ar, const unsigned long long itemId)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::CPHeader", itemId),
              colNames_(columnNames), namesProvided_(true) {}

        inline CPReference(const typename T::value_type& protoPack,
                           AbsArchive& ar, const unsigned long long itemId)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::CPHeader", itemId),
              colNames_(collectTupleNames(protoPack)), namesProvided_(true) {}

        // Compatibility constructor. It acts in the same manner
        // as the RPReference constructor: the column names will
        // be figured out from the archive, while the input types
        // must be compatible with the archived types.
        inline CPReference(AbsArchive& ar, const unsigned long long itemId)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::CPHeader", itemId),
              colNames_(), namesProvided_(false) {}

        inline CPReference(
            const std::vector<std::string>& columnNames,
            AbsArchive& ar, const SearchSpecifier& namePattern,
            const SearchSpecifier& categPattern)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::CPHeader",
                           namePattern, categPattern),
              colNames_(columnNames), namesProvided_(true) {}

        inline CPReference(
            const typename T::value_type& protoPack,
            AbsArchive& ar, const SearchSpecifier& namePattern,
            const SearchSpecifier& categPattern)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::CPHeader",
                           namePattern, categPattern),
              colNames_(collectTupleNames(protoPack)), namesProvided_(true) {}

        // Compatibility constructor
        inline CPReference(
            AbsArchive& ar, const SearchSpecifier& namePattern,
            const SearchSpecifier& categPattern)
            : AbsReference(ar, ClassId::makeId<T>(), "gs::CPHeader",
                           namePattern, categPattern),
              colNames_(), namesProvided_(false) {}

        // Disable class id comparison
        inline bool isIOCompatible(const CatalogEntry& r) const
            {return this->isSameIOPrototype(r);}

        // Methods which retrieve the object
        inline std::unique_ptr<T> get(const unsigned long index) const
            {return std::unique_ptr<T>(getPtr(index));}

        inline std::shared_ptr<T> getShared(
            const unsigned long index) const
            {return std::shared_ptr<T>(getPtr(index));}

    private:
        inline T* getPtr(const unsigned long number) const
        {
            const unsigned long long itemId = this->id(number);
            assert(itemId);
            return T::read(archive(),
                           this->positionInputStream(itemId),
                           itemId, colNames_, namesProvided_);
        }

        std::vector<std::string> colNames_;
        bool namesProvided_;
    };
}

#endif // CPP11_STD_AVAILABLE
#endif // GENERS_CPREFERENCE_HH_

