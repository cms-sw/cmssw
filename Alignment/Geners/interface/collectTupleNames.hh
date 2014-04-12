#ifndef GENERS_COLLECTTUPLENAMES_HH_
#define GENERS_COLLECTTUPLENAMES_HH_

#include <vector>
#include <string>
#include <cassert>

namespace gs {
    namespace Private 
    {
        template<typename Tuple, int N>
        struct TupleNameCycler
        {
            inline static void collect(const Tuple& t,
                                       std::vector<std::string>* n)
            {
                TupleNameCycler<Tuple, N-1>::collect(t, n);
                n->push_back((std::get<N-1>(t)).name());
            }
        };

        template<typename Tuple>
        struct TupleNameCycler<Tuple, 0>
        {
            inline static void collect(const Tuple&,
                                       std::vector<std::string>* s)
                {assert(s); s->clear();}
        };
    }

    template<typename Pack>
    std::vector<std::string> collectTupleNames(const Pack& pack)
    {
        std::vector<std::string> v;
        Private::TupleNameCycler<Pack,std::tuple_size<Pack>::value>::collect(
            pack, &v);
        return v;
    }
}

#endif // GENERS_COLLECTTUPLENAMES_HH_

