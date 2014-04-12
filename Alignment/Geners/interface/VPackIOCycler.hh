#ifndef GENERS_VPACKIOCYCLER_HH_
#define GENERS_VPACKIOCYCLER_HH_

#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    namespace Private 
    {
        template<typename Pack, int N>
        struct VPackIOCycler
        {
            template <typename Stream>
            inline static bool read(Pack* s, Stream& is)
            {
                return VPackIOCycler<Pack, N-1>::read(s, is) &&
                    process_item<GenericReader>(
                        std::get<N-1>(*(static_cast<typename Pack::Base*>(s))),
                        is, &s->iostack_[N-1], false);
            }
        };

        template<typename Pack>
        struct VPackIOCycler<Pack, 0>
        {
            template <typename Stream>
            inline static bool read(Pack*, Stream&)
                {return true;}
        };
    }
}

#endif // GENERS_VPACKIOCYCLER_HH_

