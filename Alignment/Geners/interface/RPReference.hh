#ifndef GENERS_RPREFERENCE_HH_
#define GENERS_RPREFERENCE_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <memory>

#include "Alignment/Geners/interface/AbsReference.hh"
#include "Alignment/Geners/interface/ClassId.hh"

namespace gs {
    template <typename RP>
    class RPReference : public AbsReference
    {
    public:
        inline RPReference(AbsArchive& ar, const unsigned long long itemId)
           : AbsReference(ar, ClassId::makeId<RP>(), "gs::RPHeader", itemId) {}

        inline RPReference(
            AbsArchive& ar, const SearchSpecifier& namePattern,
            const SearchSpecifier& categPattern)
            : AbsReference(ar, ClassId::makeId<RP>(), "gs::RPHeader",
                           namePattern, categPattern) {}

        inline std::unique_ptr<RP> get(const unsigned long index) const
            {return std::unique_ptr<RP>(getPtr(index));}

        inline std::shared_ptr<RP> getShared(
            const unsigned long index) const
            {return std::shared_ptr<RP>(getPtr(index));}

    private:
        inline RP* getPtr(const unsigned long number) const
        {
            const unsigned long long itemId = this->id(number);
            assert(itemId);
            return RP::read(archive(),
                            this->positionInputStream(itemId),
                            itemId);
        }
    };
}

#endif // CPP11_STD_AVAILABLE
#endif // GENERS_RPREFERENCE_HH_

