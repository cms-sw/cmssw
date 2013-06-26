#include "JetMETCorrections/InterpolationTables/interface/DualHistoAxis.h"

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"
#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"

namespace npstat {
    bool DualHistoAxis::write(std::ostream& of) const
    {
        unsigned char c = uniform_;
        gs::write_pod(of, c);
        if (uniform_)
            return !of.fail() && u_.classId().write(of) && u_.write(of);
        else
            return !of.fail() && a_.classId().write(of) && a_.write(of);
    }

    DualHistoAxis* DualHistoAxis::read(const gs::ClassId& id, std::istream& in)
    {
        static const gs::ClassId current(gs::ClassId::makeId<DualHistoAxis>());
        current.ensureSameId(id);

        unsigned char c;
        gs::read_pod(in, &c);
        gs::ClassId clid(in, 1);
        if (in.fail())
            throw gs::IOReadFailure("In npstat::DualHistoAxis::read: "
                                    "input stream failure");
        if (c)
        {
            CPP11_auto_ptr<HistoAxis> axis(HistoAxis::read(clid, in));
            return new DualHistoAxis(*axis);
        }
        else
        {
            CPP11_auto_ptr<NUHistoAxis> axis(NUHistoAxis::read(clid, in));
            return new DualHistoAxis(*axis);
        }
    }
}
