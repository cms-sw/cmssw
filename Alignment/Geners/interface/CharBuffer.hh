#ifndef GENERS_CHARBUFFER_HH_
#define GENERS_CHARBUFFER_HH_

#include <iostream>

#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/CStringBuf.hh"

namespace gs {
    class CharBuffer : public std::basic_iostream<char>
    {
    public:
        inline CharBuffer() {this->init(&buf_);}

        unsigned long size() const;

        inline ClassId classId() const {return ClassId(*this);}
        bool write(std::ostream& of) const;

        static inline const char* classname() {return "gs::CharBuffer";}
        static inline unsigned version() {return 1;}
        static void restore(const ClassId& id, std::istream& in,
                            CharBuffer* buf);

        bool operator==(const CharBuffer& r) const;
        inline bool operator!=(const CharBuffer& r) const
            {return !(*this == r);}

    private:
        CStringBuf buf_;
    };
}

#endif // GENERS_CHARBUFFER_HH_

