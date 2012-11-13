#ifndef GENERS_CPBUFFERREFERENCE_HH_
#define GENERS_CPBUFFERREFERENCE_HH_

#include <cassert>

#include "Alignment/Geners/interface/AbsReference.hh"
#include "Alignment/Geners/interface/ColumnBuffer.hh"
#include "Alignment/Geners/interface/binaryIO.hh"

namespace gs {
    namespace Private {
        class CPBufferReference : public AbsReference
        {
        public:
            inline CPBufferReference(AbsArchive& ar,
                                     const ClassId& bufClass,
                                     const ClassId& cbClass,
                                     const unsigned long long itemId)
                : AbsReference(ar, bufClass, "gs::CPBuffer", itemId),
                  bufClass_(bufClass), cbClass_(cbClass) {}

            inline void restore(const unsigned long number,
                                ColumnBuffer* obj, unsigned long* column) const
            {
                const unsigned long long itemId = this->id(number);
                assert(itemId);
                std::istream& is = this->positionInputStream(itemId);
                read_pod(is, column);
                if (is.fail()) throw IOReadFailure(
                    "In gs::Private::CPBufferReference::restore: "
                    "input stream failure");
                ColumnBuffer::restore(bufClass_, cbClass_, is, obj);
            }

        private:
            const ClassId& bufClass_;
            const ClassId& cbClass_;
        };
    }
}

#endif // GENERS_CPBUFFERREFERENCE_HH_

