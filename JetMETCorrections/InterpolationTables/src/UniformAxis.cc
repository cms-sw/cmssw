#include <cmath>
#include <climits>
#include <algorithm>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

#include "JetMETCorrections/InterpolationTables/interface/UniformAxis.h"
#include "JetMETCorrections/InterpolationTables/interface/closeWithinTolerance.h"

namespace npstat {
    UniformAxis::UniformAxis(const unsigned nCoords,
                             const double min, const double max,
                             const char* label)
        : min_(min), max_(max), label_(label ? label : ""), npt_(nCoords)
    {
        if (!(npt_ > 1U && npt_ < UINT_MAX/2U - 1U))
            throw npstat::NpstatInvalidArgument("In npstat::UniformAxis constructor: "
                                        "number of points is out of range");
        if (min_ > max_)
            std::swap(min_, max_);
        bw_ = (max_ - min_)/(npt_ - 1U);
        if (max_ == min_)
            throw npstat::NpstatInvalidArgument(
                "In npstat::UniformAxis constructor: "
                "minimum and maximum must be distinct");
    }

    std::pair<unsigned,double> UniformAxis::getInterval(const double x) const
    {
        if (x <= min_)
            return std::pair<unsigned,double>(0U, 1.0);
        else if (x >= max_)
            return std::pair<unsigned,double>(npt_ - 2U, 0.0);
        else
        {
            unsigned binnum = static_cast<unsigned>(floor((x - min_)/bw_));
            if (binnum > npt_ - 2U)
                binnum = npt_ - 2U;
            double w = binnum + 1.0 - (x - min_)/bw_;
            if (w < 0.0)
                w = 0.0;
            else if (w > 1.0)
                w = 1.0;
            return std::pair<unsigned,double>(binnum, w);
        }
    }

    std::pair<unsigned,double> UniformAxis::linearInterval(const double x) const
    {
        if (x <= min_)
            return std::pair<unsigned,double>(0U, 1.0 - (x - min_)/bw_);
        else if (x >= max_)
            return std::pair<unsigned,double>(npt_ - 2U, (max_ - x)/bw_);
        else
        {
            unsigned binnum = static_cast<unsigned>(floor((x - min_)/bw_));
            if (binnum > npt_ - 2U)
                binnum = npt_ - 2U;
            double w = binnum + 1.0 - (x - min_)/bw_;
            if (w < 0.0)
                w = 0.0;
            else if (w > 1.0)
                w = 1.0;
            return std::pair<unsigned,double>(binnum, w);
        }
    }

    std::vector<double> UniformAxis::coords() const
    {
        std::vector<double> vec;
        vec.reserve(npt_);
        const unsigned nptm1 = npt_ - 1U;
        for (unsigned i=0; i<nptm1; ++i)
            vec.push_back(min_ + bw_*i);
        vec.push_back(max_);
        return vec;
    }

    double UniformAxis::coordinate(const unsigned i) const
    {
        if (i >= npt_)
            throw npstat::NpstatOutOfRange(
                "In npstat::UniformAxis::coordinate: index out of range");
        if (i == npt_ - 1U)
            return max_;
        else
            return min_ + bw_*i;
    }

    bool UniformAxis::isClose(const UniformAxis& r, const double tol) const
    {
        return closeWithinTolerance(min_, r.min_, tol) &&
               closeWithinTolerance(max_, r.max_, tol) &&
               label_ == r.label_ &&
               npt_ == r.npt_;
    }

    bool UniformAxis::operator==(const UniformAxis& r) const
    {
        return min_ == r.min_ &&
               max_ == r.max_ &&
               label_ == r.label_ &&
               npt_ == r.npt_;
    }

    bool UniformAxis::write(std::ostream& of) const
    {
        gs::write_pod(of, min_);
        gs::write_pod(of, max_);
        gs::write_pod(of, label_);
        gs::write_pod(of, npt_);
        return !of.fail();
    }

    UniformAxis* UniformAxis::read(const gs::ClassId& id, std::istream& in)
    {
        static const gs::ClassId current(gs::ClassId::makeId<UniformAxis>());
        current.ensureSameId(id);

        double min = 0.0, max = 0.0;
        std::string label;
        unsigned nBins = 0;

        gs::read_pod(in, &min);
        gs::read_pod(in, &max);
        gs::read_pod(in, &label);
        gs::read_pod(in, &nBins);

        if (!in.fail())
            return new UniformAxis(nBins, min, max, label.c_str());
        else
            throw gs::IOReadFailure("In npstat::UniformAxis::read: "
                                    "input stream failure");
    }
}
