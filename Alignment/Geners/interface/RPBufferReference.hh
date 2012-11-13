#ifndef GENERS_RPBUFFERREFERENCE_HH_
#define GENERS_RPBUFFERREFERENCE_HH_

#include "Alignment/Geners/interface/AbsReference.hh"
#include "Alignment/Geners/interface/streamposIO.hh"

namespace gs {
    namespace Private {
        template<class Packer>
        class RPBufferReference : public AbsReference
        {
        public:
            inline RPBufferReference(const Packer& obj,
                                     const unsigned long long itemId)
                : AbsReference(obj.ar_, obj.bufferClass_, 
                               "gs::RPBuffer", itemId),
                  obj_(obj) {}

            inline void restore(const unsigned long number) const
            {
                const unsigned long long itemId = this->id(number);
                assert(itemId);
                std::istream& is = this->positionInputStream(itemId);
                read_pod(is, &obj_.firstReadBufferRow_);
                read_pod_vector(is, &obj_.readBufferOffsets_);
                CharBuffer::restore(obj_.bufferClass_, is, &obj_.readBuffer_);
            }

        private:
            const Packer& obj_;
        };
    }
}

#endif // GENERS_RPBUFFERREFERENCE_HH_

