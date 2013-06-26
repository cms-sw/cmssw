#ifndef GENERS_COLUMNPACKERHELPER_HH_
#define GENERS_COLUMNPACKERHELPER_HH_

#include <tuple>

#include "Alignment/Geners/interface/GenericIO.hh"
#include "Alignment/Geners/interface/IOReferredType.hh"
#include "Alignment/Geners/interface/IOIsPOD.hh"
#include "Alignment/Geners/interface/ColumnBuffer.hh"

namespace gs {
    template<typename Pack> class ColumnPacker;

    namespace Private {
        template<typename Pack, unsigned long N>
        struct ColumnPackerHelper
        {
            typedef typename IOReferredType<
                typename std::tuple_element<N-1,Pack>::type>::type 
            element_io_type;

            inline static void podness(
                std::vector<Private::ColumnBuffer*>& buf)
            {
                ColumnPackerHelper<Pack, N-1>::podness(buf);
                if (IOIsPOD<element_io_type>::value)
                    buf[N-1]->podsize = sizeof(element_io_type);
            }

            inline static bool write(
                ColumnPacker<Pack>& packer, const Pack& pack)
            {
                const bool status = ColumnPackerHelper<Pack, N-1>::write(
                    packer, pack);
                std::ostream& os = packer.columnOstream(N-1);
                char* ps = 0;
                return status && process_const_item<GenericWriter>(
                    std::get<N-1>(pack), os, ps, false);
            }

            inline static bool readRow(
                const ColumnPacker<Pack>& packer, Pack* pack)
            {
                if (!ColumnPackerHelper<Pack, N-1>::readRow(packer, pack))
                    return false;
                std::vector<ClassId>* iostack = 0;
                std::istream* is = packer.columnIstream(N-1, &iostack);
                if (is)
                    return process_item<GenericReader>(
                        std::get<N-1>(*pack), *is, iostack, false);
                else
                    // It is OK to skip a column on readout
                    return true;
            }
        };

        template<typename Pack>
        struct ColumnPackerHelper<Pack, 0UL>
        {
            inline static void podness(std::vector<Private::ColumnBuffer*>&) {}
            inline static bool write(ColumnPacker<Pack>&, const Pack&)
                {return true;}
            inline static bool readRow(const ColumnPacker<Pack>&, Pack*)
                {return true;}
        };
    }
}

#endif // GENERS_COLUMNPACKERHELPER_HH_

