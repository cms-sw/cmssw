#ifndef GENERS_COLUMNBUFFER_HH_
#define GENERS_COLUMNBUFFER_HH_

#include <vector>
#include <iostream>

#include "Alignment/Geners/interface/CharBuffer.hh"
#include "Alignment/Geners/interface/ClassId.hh"

namespace gs {
    namespace Private {
        struct ColumnBuffer
        {
            inline ColumnBuffer() : firstrow(0), lastrowp1(0), podsize(0) {}

            // The actual buffer
            CharBuffer buf;

            // The range of rows stored in the buffer is [firstrow, lastrowp1)
            unsigned long firstrow;
            unsigned long lastrowp1;

            // The following member will be non-0 if the column
            // members are PODs for I/O purposes
            unsigned long podsize;

            // If the above member is 0, objects in the buffer may have
            // variable size. Because of this, we need to store an offset
            // of each object separately. To conserve space, the vector
            // of object offsets is not filled for PODs.
            std::vector<std::streampos> offsets;

            // Methods related to I/O
            bool write(std::ostream& os) const;
            inline const ClassId& classId() const
            {
                return static_classId();
            }

            // The "restore" method is special and requires
            // use of special references. The reason is the
            // following: we will likely store a large number
            // of these buffers at a time. Storing the class id
            // of CharBuffer every time would be a waste of
            // storage space. Because of this, we will assume
            // that the code that manages these buffers will
            // stash the CharBuffer class id somewhere.
            static void restore(const ClassId& id, const ClassId& charBufId,
                                std::istream& in, ColumnBuffer* buf);

            static inline const char* classname()
                {return "gs::Private::ColumnBuffer";}
            static inline unsigned version() {return 1;}

            bool operator==(const ColumnBuffer& r) const;
            inline bool operator!=(const ColumnBuffer& r) const
                {return !(*this == r);}

        private:
            static const ClassId& static_classId();
        };
    }
}

#endif // GENERS_COLUMNBUFFER_HH_

