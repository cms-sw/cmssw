#ifndef GENERS_INT2TYPE_HH_
#define GENERS_INT2TYPE_HH_

#include <sstream>

namespace gs {
    namespace Private {
        inline std::string makeInt2TypeStage(const int v)
        {
            std::ostringstream os;
            os << "Int2Type::" << v;
            return os.str();
        }
    }

    template <int v>
    struct Int2Type
    {
        enum {value = v};

        static const char* stage()
        {
            static const std::string buf(Private::makeInt2TypeStage(v));
            return buf.c_str();
        }
    };
}

#endif // GENERS_INT2TYPE_HH_

