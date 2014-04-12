#include <cmath>
#include <algorithm>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

#include "JetMETCorrections/InterpolationTables/interface/GridAxis.h"
#include "JetMETCorrections/InterpolationTables/interface/closeWithinTolerance.h"

namespace npstat {
    void GridAxis::initialize()
    {
        if (npt_ <= 1U) throw npstat::NpstatInvalidArgument(
            "In npstat::GridAxis::initialize: insufficient number "
            "of coordinates (at least 2 are required)");
        std::sort(coords_.begin(), coords_.end());
        const double* c = &coords_[0];
        if (useLogSpace_)
        {
            logs_.reserve(npt_);
            for (unsigned i=0; i<npt_; ++i)
            {
                if (c[i] <= 0.0) throw npstat::NpstatInvalidArgument(
                    "In npstat::GridAxis::initialize: in log space"
                    "all coordinates must be positive");
                logs_.push_back(log(c[i]));
            }
        }

        // Can't have duplicate coordinates
        for (unsigned i=1; i<npt_; ++i)
            if (c[i] <= c[i - 1U]) throw npstat::NpstatInvalidArgument(
                "In npstat::GridAxis::initialize: "
                "all coordinates must be distinct");
    }

    GridAxis::GridAxis(const std::vector<double>& coords,
                       const bool useLogSpace)
        : coords_(coords),
          npt_(coords_.size()),
          useLogSpace_(useLogSpace)
    {
        initialize();
    }

    GridAxis::GridAxis(const std::vector<double>& coords,
                       const char* label,
                       const bool useLogSpace)
        : coords_(coords),
          label_(label ? label : ""),
          npt_(coords_.size()),
          useLogSpace_(useLogSpace)
    {
        initialize();
    }

    std::pair<unsigned,double> GridAxis::getInterval(const double x) const
    {
        if (useLogSpace_)
            if (x <= 0.0) throw npstat::NpstatDomainError(
                "In GridAxis::getInterval: argument must be positive");
        const double* c = &coords_[0];
        if (x <= c[0])
            return std::pair<unsigned,double>(0U, 1.0);
        else if (x >= c[npt_ - 1U])
            return std::pair<unsigned,double>(npt_ - 2U, 0.0);
        else
        {
            unsigned ibnd = lower_bound
                (coords_.begin(), coords_.end(), x) - coords_.begin();
            --ibnd;
            if (useLogSpace_)
            {
                const double* l = &logs_[0];
                const double w = 1.0 - (log(x) - l[ibnd])/
                    (l[ibnd + 1U] - l[ibnd]);
                return std::pair<unsigned,double>(ibnd, w);
            }
            else
            {
                const double w = 1.0 - (x - c[ibnd])/(c[ibnd + 1U] - c[ibnd]);
                return std::pair<unsigned,double>(ibnd, w);
            }
        }
    }

    std::pair<unsigned,double> GridAxis::linearInterval(const double x) const
    {
        if (useLogSpace_)
            if (x <= 0.0) throw npstat::NpstatDomainError(
                "In GridAxis::linearInterval: argument must be positive");
        const double* c = &coords_[0];
        if (x <= c[0])
        {
            if (useLogSpace_)   
            {
                const double* l = &logs_[0];
                const double bw = l[1] - l[0];
                return std::pair<unsigned,double>(0U, 1.0 - (log(x) - l[0])/bw);
            }
            else
            {
                const double bw = c[1] - c[0];
                return std::pair<unsigned,double>(0U, 1.0 - (x - c[0])/bw);
            }
        }
        else if (x >= c[npt_ - 1U])
        {
            if (useLogSpace_)
            {
                const double* l = &logs_[0];
                const double bw = l[npt_ - 1U] - l[npt_ - 2U];
                return std::pair<unsigned,double>(
                    npt_ - 2U, (l[npt_ - 1U] - log(x))/bw);
            }
            else
            {
                const double bw = c[npt_ - 1U] - c[npt_ - 2U];
                return std::pair<unsigned,double>(
                    npt_ - 2U, (c[npt_ - 1U] - x)/bw);
            }
        }
        else
        {
            unsigned ibnd = lower_bound
                (coords_.begin(), coords_.end(), x) - coords_.begin();
            --ibnd;
            if (useLogSpace_)
            {
                const double* l = &logs_[0];
                const double w = 1.0 - (log(x) - l[ibnd])/
                    (l[ibnd + 1U] - l[ibnd]);
                return std::pair<unsigned,double>(ibnd, w);
            }
            else
            {
                const double w = 1.0 - (x - c[ibnd])/(c[ibnd + 1U] - c[ibnd]);
                return std::pair<unsigned,double>(ibnd, w);
            }
        }
    }

    bool GridAxis::write(std::ostream& of) const
    {
        // It is unlikely that this object will be written in isolation.
        // So, do not bother with too many checks.
        gs::write_pod_vector(of, coords_);
        gs::write_pod(of, label_);
        gs::write_pod(of, useLogSpace_);
        return !of.fail();        
    }

    GridAxis* GridAxis::read(const gs::ClassId& id, std::istream& in)
    {
        static const gs::ClassId current(gs::ClassId::makeId<GridAxis>());
        current.ensureSameId(id);

        std::vector<double> coords;
        gs::read_pod_vector(in, &coords);

        std::string label;
        gs::read_pod(in, &label);

        bool useLogSpace;
        gs::read_pod(in, &useLogSpace);

        if (in.fail())
            throw gs::IOReadFailure("In npstat::GridAxis::read: "
                                    "input stream failure");
        return new GridAxis(coords, label.c_str(), useLogSpace);
    }

    bool GridAxis::operator==(const GridAxis& r) const
    {
        return useLogSpace_ == r.useLogSpace_ &&
               coords_ == r.coords_ &&
               label_ == r.label_;
    }

    bool GridAxis::isClose(const GridAxis& r, const double tol) const
    {
        if (!(useLogSpace_ == r.useLogSpace_ &&
              label_ == r.label_))
            return false;
        const unsigned long n = coords_.size();
        if (n != r.coords_.size())
            return false;
        for (unsigned long i=0; i<n; ++i)
            if (!closeWithinTolerance(coords_[i], r.coords_[i], tol))
                return false;
        return true;
    }
}
