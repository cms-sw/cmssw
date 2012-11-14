#include <cmath>
#include <climits>
#include <algorithm>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

#include "JetMETCorrections/InterpolationTables/interface/closeWithinTolerance.h"
#include "JetMETCorrections/InterpolationTables/interface/HistoAxis.h"

namespace npstat {
    HistoAxis::HistoAxis(const unsigned nbins, const double min,
                         const double max, const char* label)
        : min_(min), max_(max), label_(label ? label : ""), nBins_(nbins)
    {
        if (!(nBins_ && nBins_ < UINT_MAX/2U - 1U))
            throw npstat::NpstatInvalidArgument("In npstat::HistoAxis constructor: "
                                        "number of bins is out of range");
        if (min_ > max_)
            std::swap(min_, max_);
        bw_ = (max_ - min_)/nBins_;
    }

    bool HistoAxis::isClose(const HistoAxis& r, const double tol) const
    {
        return closeWithinTolerance(min_, r.min_, tol) &&
               closeWithinTolerance(max_, r.max_, tol) &&
               label_ == r.label_ &&
               nBins_ == r.nBins_;
    }

    bool HistoAxis::operator==(const HistoAxis& r) const
    {
        return min_ == r.min_ &&
               max_ == r.max_ &&
               label_ == r.label_ &&
               nBins_ == r.nBins_;
    }

    bool HistoAxis::operator!=(const HistoAxis& r) const
    {
        return !(*this == r);
    }

    int HistoAxis::binNumber(const double x) const
    {
        if (bw_)
        {
            int binnum = static_cast<int>(floor((x - min_)/bw_));
            if (x >= max_)
            {
                if (binnum < static_cast<int>(nBins_))
                    binnum = nBins_;
            }
            else
            {
                if (binnum >= static_cast<int>(nBins_))
                    binnum = nBins_ - 1U;
            }
            return binnum;
        }
        else
        {
            if (x < min_)
                return -1;
            else
                return nBins_;
        }
    }

    unsigned HistoAxis::closestValidBin(const double x) const
    {
        if (x <= min_)
            return 0U;
        else if (bw_ && x < max_)
        {
            const unsigned binnum = static_cast<unsigned>(floor((x-min_)/bw_));
            if (binnum < nBins_)
                return binnum;
        }
        return nBins_ - 1U;
    }

    LinearMapper1d HistoAxis::binNumberMapper(const bool mapLeftEdgeTo0) const
    {
        if (!bw_) throw npstat::NpstatDomainError(
            "In npstat::HistoAxis::binNumberMapper: "
            "bin width is zero. Mapper can not be constructed.");
        const double base = mapLeftEdgeTo0 ? min_/bw_ : min_/bw_ + 0.5;
        return LinearMapper1d(1.0/bw_, -base);
    }

    CircularMapper1d HistoAxis::kernelScanMapper(const bool doubleRange) const
    {
        if (!bw_) throw npstat::NpstatDomainError(
            "In npstat::HistoAxis::kernelScanMapper: "
            "bin width is zero. Mapper can not be constructed.");
        double range = max_ - min_;
        if (doubleRange)
            range *= 2.0;
        return CircularMapper1d(bw_, 0.0, range);
    }

    unsigned HistoAxis::overflowIndexWeighted(
        const double x, unsigned* binNumber, double *weight) const
    {
        if (x < min_)
            return 0U;
        else if (x >= max_)
            return 2U;
        else
        {
            if (nBins_ <= 1U) throw npstat::NpstatInvalidArgument(
                "In npstat::HistoAxis::overflowIndexWeighted: "
                "must have more than one bin");
            const double dbin = (x - min_)/bw_;
            if (dbin <= 0.5)
            {
                *binNumber = 0;
                *weight = 1.0;
            }
            else if (dbin >= nBins_ - 0.5)
            {
                *binNumber = nBins_ - 2;
                *weight = 0.0;
            }
            else
            {
                const unsigned bin = static_cast<unsigned>(dbin - 0.5);
                *binNumber = bin >= nBins_ - 1U ? nBins_ - 2U : bin;
                *weight = 1.0 - (dbin - 0.5 - *binNumber);
            }
            return 1U;
        }
    }

    bool HistoAxis::write(std::ostream& of) const
    {
        gs::write_pod(of, min_);
        gs::write_pod(of, max_);
        gs::write_pod(of, label_);
        gs::write_pod(of, nBins_);
        return !of.fail();
    }

    HistoAxis* HistoAxis::read(const gs::ClassId& id, std::istream& in)
    {
        static const gs::ClassId current(gs::ClassId::makeId<HistoAxis>());
        current.ensureSameId(id);

        double min = 0.0, max = 0.0;
        std::string label;
        unsigned nBins = 0;

        gs::read_pod(in, &min);
        gs::read_pod(in, &max);
        gs::read_pod(in, &label);
        gs::read_pod(in, &nBins);

        if (!in.fail())
            return new HistoAxis(nBins, min, max, label.c_str());
        else
            throw gs::IOReadFailure("In npstat::HistoAxis::read: "
                                    "input stream failure");
    }
}
