#ifndef GENERS_CPFOOTERREFERENCE_HH_
#define GENERS_CPFOOTERREFERENCE_HH_

#include "Alignment/Geners/interface/AbsReference.hh"
#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    namespace Private {
        struct CPFooterReference : public AbsReference
        {
            inline CPFooterReference(
                AbsArchive& ar, const ClassId& classId,
                const char* name, const char* category)
                : AbsReference(ar, classId, "gs::CPFooter",
                               name, category) {}

            // Disable class id comparison
            inline bool isIOCompatible(const CatalogEntry& r) const
                {return this->isSameIOPrototype(r);}

            void fillItems(
                unsigned long* nrows, unsigned long long* headerId,
                std::vector<std::vector<
                    std::pair<unsigned long,unsigned long long> > >* ids,
                unsigned long long* offset,
                const unsigned long number) const
            {
                const unsigned long long itemId = this->id(number);
                assert(itemId);
                std::istream& s = this->positionInputStream(itemId);
                *offset = archive().catalogEntry(itemId)->offset();
                read_pod(s, nrows);
                read_pod(s, headerId);
                restore_item(s, ids, false);
            }
        };
    }
}

#endif // GENERS_CPFOOTERREFERENCE_HH_

