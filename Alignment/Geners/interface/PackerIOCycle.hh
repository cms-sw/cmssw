#ifndef GENERS_PACKERIOCYCLE_HH_
#define GENERS_PACKERIOCYCLE_HH_

#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    namespace Private {
        // Before calling this, make sure that iostack is properly filled
        template<typename Pack, unsigned long N>
        struct PackerIOCycle
        {
            template <typename Stream>
            inline static bool read(
                Pack* s, Stream& is,
                std::vector<std::vector<ClassId> >& iostack)
            {
                return PackerIOCycle<Pack, N-1>::read(s, is, iostack) &&
                    process_item<GenericReader>(
                        std::get<N-1>(*s), is, &iostack[N-1], false);
            }
        };

        template<typename Pack>
        struct PackerIOCycle<Pack, 0UL>
        {
            template <typename Stream>
            inline static bool read(Pack*, Stream&,
                                    std::vector<std::vector<ClassId> >&)
                {return true;}
        };
    }
}

#endif // GENERS_PACKERIOCYCLE_HH_

