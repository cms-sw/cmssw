#include "Alignment/Geners/interface/ColumnBuffer.hh"
#include "Alignment/Geners/interface/streamposIO.hh"

namespace gs {
    namespace Private {
        bool ColumnBuffer::write(std::ostream& os) const
        {
            write_pod(os, firstrow);
            write_pod(os, lastrowp1);
            const unsigned char isPod = (podsize ? 1 : 0);
            write_pod(os, isPod);
            if (isPod)
                write_pod(os, podsize);
            else
                write_pod_vector(os, offsets);
            return !os.fail() && buf.write(os);
        }

        void ColumnBuffer::restore(const ClassId& id, const ClassId& bufId,
                                   std::istream& is, ColumnBuffer* obj)
        {
            assert(obj);
            obj->classId().ensureSameId(id);

            read_pod(is, &obj->firstrow);
            read_pod(is, &obj->lastrowp1);
            unsigned char isPod;
            read_pod(is, &isPod);
            if (isPod)
            {
                obj->offsets.clear();
                read_pod(is, &obj->podsize);
            }
            else
            {
                obj->podsize = 0;
                read_pod_vector(is, &obj->offsets);
            }
            if (is.fail()) throw IOReadFailure(
                "In gs::Private::ColumnBuffer::restore: input stream failure");

            // Size of a POD object can not be zero
            if (isPod && !obj->podsize) throw IOInvalidData(
                "In gs::Private::ColumnBuffer::restore: corrupted record");

            CharBuffer::restore(bufId, is, &obj->buf);
        }

        bool ColumnBuffer::operator==(const ColumnBuffer& r) const
        {
            return firstrow == r.firstrow &&
                   lastrowp1 == r.lastrowp1 &&
                   podsize == r.podsize &&
                   offsets == r.offsets &&
                   buf == r.buf;
        }

        const ClassId& ColumnBuffer::static_classId()
        {
            static const ClassId classId_(ClassId(classname(), version()));
            return classId_;
        }
    }
}
