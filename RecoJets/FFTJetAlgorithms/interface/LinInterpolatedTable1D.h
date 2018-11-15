//=========================================================================
// LinInterpolatedTable1D.h
//
// This class can be used to linearly interpolate data in one dimension.
// It differs from similar facilities in the fftjet package by its handling
// of the extrapolation beyound the grid limits.
//
// I. Volobouev
// April 2011
//=========================================================================

#ifndef RecoJets_FFTJetAlgorithms_LinInterpolatedTable1D_h
#define RecoJets_FFTJetAlgorithms_LinInterpolatedTable1D_h

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

#include "fftjet/SimpleFunctors.hh"
#include "FWCore/Utilities/interface/Exception.h"

namespace fftjetcms {
    class LinInterpolatedTable1D : public fftjet::Functor1<double,double>
    {
    public:
        // Constructor from a regularly spaced data. Extrapolation
        // from the edge to infinity can be either linear or constant.
        // "npoints" must be larger than 1.
        template <typename RealN>
        LinInterpolatedTable1D(const RealN* data, unsigned npoints,
                               double x_min, double x_max,
                               bool leftExtrapolationLinear,
                               bool rightExtrapolationLinear);

        // Constructor from a list of points, not necessarily regularly
        // spaced (but must be sorted in the increasing order). The first
        // member of the pair is the x coordinate, the second is the
        // tabulated function value. The input list will be interpolated
        // to "npoints" internal points linearly.
        template <typename RealN>
        LinInterpolatedTable1D(const std::vector<std::pair<RealN,RealN> >& v,
                               unsigned npoints,
                               bool leftExtrapolationLinear,
                               bool rightExtrapolationLinear);

        // Constructor which builds a function returning the given constant
        explicit LinInterpolatedTable1D(double c);

        inline ~LinInterpolatedTable1D() override {}

        // Main calculations are performed inside the following operator
        double operator()(const double& x) const override;

        // Comparisons (useful for testing)
        bool operator==(const LinInterpolatedTable1D& r) const;
        inline bool operator!=(const LinInterpolatedTable1D& r) const
            {return !(*this == r);}

        // Various simple inspectors
        inline double xmin() const {return xmin_;}
        inline double xmax() const {return xmax_;}
        inline unsigned npoints() const {return npoints_;}
        inline bool leftExtrapolationLinear() const
            {return leftExtrapolationLinear_;}
        inline bool rightExtrapolationLinear() const
            {return rightExtrapolationLinear_;}
        inline const double* data() const {return &data_[0];}

        // The following checks whether the table is monotonous
        // (and, therefore, can be inverted). Possible flat regions
        // at the edges are not taken into account.
        bool isMonotonous() const;

        // Generate the inverse lookup table. Note that it is only
        // possible if the original table is monotonous (not taking
        // into account possible flat regions at the edges). If the
        // inversion is not possible, NULL pointer will be returned.
        //
        // The new table will have "npoints" points. The parameters
        // "leftExtrapolationLinear" and "rightExtrapolationLinear"
        // refer to the inverted table (note that left and right will
        // exchange places if the original table is decreasing).
        //
        std::unique_ptr<LinInterpolatedTable1D> inverse(
            unsigned npoints, bool leftExtrapolationLinear,
            bool rightExtrapolationLinear) const;

    private:
        static inline double interpolateSimple(
            const double x0, const double x1,
            const double y0, const double y1,
            const double x)
        {
            return y0 + (y1 - y0)*((x - x0)/(x1 - x0));
        }

        std::vector<double> data_;
        double xmin_;
        double xmax_;
        double binwidth_;
        unsigned npoints_;
        bool leftExtrapolationLinear_;
        bool rightExtrapolationLinear_;
        mutable bool monotonous_;
        mutable bool monotonicityKnown_;
    };
}


// Implementation of the templated constructors
namespace fftjetcms {
    template <typename RealN>
    inline LinInterpolatedTable1D::LinInterpolatedTable1D(
        const RealN* data, const unsigned npoints,
        const double x_min, const double x_max,
        const bool leftExtrapolationLinear,
        const bool rightExtrapolationLinear)
        : data_(data, data+npoints),
          xmin_(x_min),
          xmax_(x_max),
          binwidth_((x_max - x_min)/(npoints - 1U)),
          npoints_(npoints),
          leftExtrapolationLinear_(leftExtrapolationLinear),
          rightExtrapolationLinear_(rightExtrapolationLinear),
          monotonous_(false),
          monotonicityKnown_(false)
    {
        if (!data)
            throw cms::Exception("FFTJetBadConfig")
                << "No data configured" << std::endl;
        if (npoints <= 1U)
            throw cms::Exception("FFTJetBadConfig")
                << "Not enough data points" << std::endl;
    }

    template <typename RealN>
    LinInterpolatedTable1D::LinInterpolatedTable1D(
        const std::vector<std::pair<RealN,RealN> >& v,
        const unsigned npoints,
        const bool leftExtrapolationLinear,
        const bool rightExtrapolationLinear)
        : xmin_(v[0].first),
          xmax_(v[v.size() - 1U].first),
          binwidth_((xmax_ - xmin_)/(npoints - 1U)),
          npoints_(npoints),
          leftExtrapolationLinear_(leftExtrapolationLinear),
          rightExtrapolationLinear_(rightExtrapolationLinear),
          monotonous_(false),
          monotonicityKnown_(false)
    {
        const unsigned len = v.size();
        if (len <= 1U)
            throw cms::Exception("FFTJetBadConfig")
                << "Not enough data for interpolation"
                << std::endl;

        if (npoints <= 1U)
            throw cms::Exception("FFTJetBadConfig")
                << "Not enough interpolation table entries"
                << std::endl;

        const std::pair<RealN,RealN>* vdata = &v[0];
        for (unsigned i=1; i<len; ++i)
            if (vdata[i-1U].first >= vdata[i].first)
                throw cms::Exception("FFTJetBadConfig")
                    << "Input data is not sorted properly"
                    << std::endl;

        unsigned shift = 0U;
        if (leftExtrapolationLinear)
        {
            ++npoints_;
            xmin_ -= binwidth_;
            shift = 1U;
        }
        if (rightExtrapolationLinear)
        {
            ++npoints_;
            xmax_ += binwidth_;
        }

        data_.resize(npoints_);

        if (leftExtrapolationLinear)
        {
            data_[0] = interpolateSimple(
                vdata[0].first, vdata[1].first,
                vdata[0].second, vdata[1].second, xmin_);
        }
        if (rightExtrapolationLinear)
        {
            data_[npoints_-1U] = interpolateSimple(
                vdata[len - 2U].first, vdata[len - 1U].first,
                vdata[len - 2U].second, vdata[len - 1U].second, xmax_);
        }

        data_[shift] = vdata[0].second;
        data_[npoints - 1U + shift] = vdata[len - 1U].second;
        unsigned ibelow = 0, iabove = 1;
        for (unsigned i=1; i<npoints-1; ++i)
        {
            const double x = xmin_ + (i + shift)*binwidth_;
            while (static_cast<double>(v[iabove].first) <= x)
            {
                ++ibelow;
                ++iabove;
            }
            if (v[ibelow].first == v[iabove].first)
                data_[i + shift] = (v[ibelow].second + v[iabove].second)/2.0;
            else
                data_[i + shift] = interpolateSimple(
                    v[ibelow].first, v[iabove].first,
                    v[ibelow].second, v[iabove].second, x);
        }
    }
}

#endif // RecoJets_FFTJetAlgorithms_LinInterpolatedTable1D_h
