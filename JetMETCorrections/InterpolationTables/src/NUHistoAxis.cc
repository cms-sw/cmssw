#include <cmath>
#include <cassert>
#include <climits>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"
#include <algorithm>

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"
#include "JetMETCorrections/InterpolationTables/interface/NUHistoAxis.h"

#include "JetMETCorrections/InterpolationTables/interface/EquidistantSequence.h"
#include "JetMETCorrections/InterpolationTables/interface/closeWithinTolerance.h"

namespace npstat {
    NUHistoAxis::NUHistoAxis(const std::vector<double>& binEdges,
                             const char* label)
        : binEdges_(binEdges), nBins_(binEdges.size() - 1U), uniform_(false)
    {
        if (!(binEdges_.size() > 1U && binEdges_.size() < UINT_MAX/2U))
            throw npstat::NpstatInvalidArgument("In npstat::NUHistoAxis constructor: "
                                        "number of bin edges is out of range");
        std::sort(binEdges_.begin(), binEdges_.end());
        min_ = binEdges_[0];
        max_ = binEdges_[nBins_];
        if (label)
            label_ = std::string(label);
    }

    NUHistoAxis::NUHistoAxis(const unsigned nBins,
                             const double min, const double max,
                             const char* label)
        : min_(min), max_(max), nBins_(nBins), uniform_(true)
    {
        if (!(nBins_ && nBins_ < UINT_MAX/2U - 1U))
            throw npstat::NpstatInvalidArgument("In npstat::NUHistoAxis constructor: "
                                        "number of bins is out of range");
        if (min_ > max_)
            std::swap(min_, max_);
        binEdges_ = EquidistantInLinearSpace(min_, max_, nBins+1U);
        if (label)
            label_ = std::string(label);
    }

    bool NUHistoAxis::isClose(const NUHistoAxis& r, const double tol) const
    {
        if (!(closeWithinTolerance(min_, r.min_, tol) && 
              closeWithinTolerance(max_, r.max_, tol) &&
              label_ == r.label_ &&
              nBins_ == r.nBins_ &&
              uniform_ == r.uniform_))
            return false;
        for (unsigned i=0; i<nBins_; ++i)
            if (!closeWithinTolerance(binEdges_[i], r.binEdges_[i], tol))
                return false;
        return true;
    }

    bool NUHistoAxis::operator==(const NUHistoAxis& r) const
    {
        return min_ == r.min_ && 
               max_ == r.max_ && 
               label_ == r.label_ &&
               nBins_ == r.nBins_ &&
               binEdges_ == r.binEdges_ &&
               uniform_ == r.uniform_;
    }

    bool NUHistoAxis::operator!=(const NUHistoAxis& r) const
    {
        return !(*this == r);
    }

    int NUHistoAxis::binNumber(const double x) const
    {
        const int delta = std::upper_bound(binEdges_.begin(), binEdges_.end(), x) - 
                          binEdges_.begin();
        return delta - 1;
    }

    double NUHistoAxis::fltBinNumber(const double x, const bool mapLeftEdgeTo0) const
    {
        const int delta = std::upper_bound(binEdges_.begin(), binEdges_.end(), x) - 
                          binEdges_.begin();
        const int binnum = delta - 1;

        if (binnum < 0)
        {
            const double left = binEdges_[0];
            const double right = binEdges_[1];
            double bval = (x - left)/(right - left);
            if (!mapLeftEdgeTo0)
                bval -= 0.5;
            if (bval < -1.0)
                bval = -1.0;
            return bval;
        }
        else if (static_cast<unsigned>(binnum) >= nBins_)
        {
            const double left = binEdges_[nBins_ - 1U];
            const double right = binEdges_[nBins_];
            double bval = nBins_ - 1U + (x - left)/(right - left);
            if (!mapLeftEdgeTo0)
                bval -= 0.5;
            if (bval > static_cast<double>(nBins_))
                bval = nBins_;
            return bval;
        }
        else
        {
            const double left = binEdges_[binnum];
            const double right = binEdges_[delta];
            if (mapLeftEdgeTo0)
                return binnum + (x - left)/(right - left);
            else
            {
                // Bin center is mapped to binnum.
                // Bin center of the next bin is mapped to binnum + 1.
                // Bin center of the previos bin is mapped to binnum - 1.
                const double binCenter = (left + right)/2.0;
                if ((binnum == 0 && x <= binCenter) ||
                    (static_cast<unsigned>(binnum) == nBins_ - 1 && x >= binCenter))
                    return binnum + (x - left)/(right - left) - 0.5;
                else if (x <= binCenter)
                {
                    const double otherBinCenter = (left + binEdges_[binnum - 1])/2.0;
                    return binnum + (x - binCenter)/(binCenter - otherBinCenter);
                }
                else
                {
                    const double otherBinCenter = (right + binEdges_[binnum + 1])/2.0;
                    return binnum + (x - binCenter)/(otherBinCenter - binCenter);
                }
            }
        }
    }

    unsigned NUHistoAxis::closestValidBin(const double x) const
    {
        const int delta = std::upper_bound(binEdges_.begin(), binEdges_.end(), x) - 
                          binEdges_.begin();
        int binnum = delta - 1;
        if (binnum < 0)
            binnum = 0;
        else if (static_cast<unsigned>(binnum) >= nBins_)
            binnum = nBins_ - 1U;
        return binnum;
    }

    bool NUHistoAxis::write(std::ostream& of) const
    {
        gs::write_pod_vector(of, binEdges_);
        gs::write_pod(of, label_);
        unsigned char c = uniform_;
        gs::write_pod(of, c);
        return !of.fail();
    }

    NUHistoAxis* NUHistoAxis::read(const gs::ClassId& id, std::istream& in)
    {
        static const gs::ClassId current(gs::ClassId::makeId<NUHistoAxis>());
        current.ensureSameId(id);

        std::vector<double> binEdges;
        std::string label;
        unsigned char unif;
        gs::read_pod_vector(in, &binEdges);
        gs::read_pod(in, &label);
        gs::read_pod(in, &unif);
        if (in.fail())
            throw gs::IOReadFailure("In npstat::UHistoAxis::read: "
                                    "input stream failure");
        NUHistoAxis* result = new NUHistoAxis(binEdges, label.c_str());
        result->uniform_ = unif;
        return result;
    }
}
