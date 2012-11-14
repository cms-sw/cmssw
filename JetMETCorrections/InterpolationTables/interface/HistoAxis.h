#ifndef NPSTAT_HISTOAXIS_HH_
#define NPSTAT_HISTOAXIS_HH_

/*!
// \file HistoAxis.h
//
// \brief Histogram axis with equidistant bins
//
// Author: I. Volobouev
//
// July 2010
*/

#include <utility>

#include "Alignment/Geners/interface/ClassId.hh"

#include "JetMETCorrections/InterpolationTables/interface/CircularMapper1d.h"
#include "JetMETCorrections/InterpolationTables/interface/Interval.h"

namespace npstat {
    template <typename Numeric, class Axis> class HistoND;
    class DualHistoAxis;

    /**
    // Class which contain the information needed to define a histogram axis.
    // All bins will have the same width. See NUHistoAxis and DualHistoAxis
    // classes for non-uniform binning.
    */
    class HistoAxis
    {
    public:
        /**
        // Minimum and maximum will be internally swapped
        // if the minimum parameter is larger than the maximum
        */
        HistoAxis(unsigned nBins, double min, double max,
                  const char* label=0);

        //@{
        /** Examine axis properties */
        inline double min() const {return min_;}
        inline double max() const {return max_;}
        inline Interval<double> interval() const
            {return Interval<double>(min_, max_);}
        inline double length() const {return max_ - min_;}
        inline unsigned nBins() const {return nBins_;}
        inline double binWidth(const int /*binNum*/=0) const {return bw_;}
        inline const std::string& label() const {return label_;}
        inline bool isUniform() const {return true;}
        //@}

        /** Return the coordinate of the given bin center */
        inline double binCenter(const int binNum) const
            {return min_ + (binNum + 0.5)*bw_;}

        /** Return the coordinate of the given bin left edge */
        inline double leftBinEdge(const int binNum) const
            {return min_ + binNum*bw_;}

        /** Return the coordinate of the given bin right edge */
        inline double rightBinEdge(const int binNum) const
            {return min_ + (binNum + 1)*bw_;}

        /** Return the coordinate interval occupied by the given bin */
        inline Interval<double> binInterval(const int binNum) const
            {return Interval<double>(min_+binNum*bw_, min_+(binNum+1)*bw_);}

        /** Change the axis label */
        inline void setLabel(const char* newlabel)
            {label_ = newlabel ? newlabel : "";}

        /**
        // This method returns arbitrary integer bin number, including
        // negative numbers and numbers which can exceed nBins()-1
        */
        int binNumber(double x) const;

        /**
        // This method returns the closest valid bin number
        // (above 0 and below nBins() )
        */
        unsigned closestValidBin(double x) const;

        /**
        // Return the mapper which calculates floating point bin number
        // given the coordinate. The resulting bin number can go above
        // and below the axis range. If "mapLeftEdgeTo0" is specified
        // as "false", it is the center of the first bin which gets
        // mapped to 0.
        */
        LinearMapper1d binNumberMapper(bool mapLeftEdgeTo0=true) const;

        /**
        // Floating point bin number given the coordinate (no bin number
        // truncation of any kind is performed). Works in exactly the same
        // way as the mapper returned by the previous method.
        */
        inline double fltBinNumber(const double x,
                                   const bool mapLeftEdgeTo0=true) const
            {return (x - min_)/bw_ - (mapLeftEdgeTo0 ? 0.0 : 0.5);}

        /**
        // The following function returns a mapper that can be
        // helpful in scanning a kernel (a density function) for
        // subsequent convolution with the histogram which contains
        // this axis.
        */
        CircularMapper1d kernelScanMapper(bool doubleRange) const;

        bool operator==(const HistoAxis&) const;
        bool operator!=(const HistoAxis&) const;

        /** Comparison of axis coordinates within given tolerance */
        bool isClose(const HistoAxis&, double tol) const;

        //@{
        /** Method related to "geners" I/O */
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static inline const char* classname() {return "npstat::HistoAxis";}
        static inline unsigned version() {return 1;}
        static HistoAxis* read(const gs::ClassId& id, std::istream& in);

    private:
        inline HistoAxis() : min_(0.0), max_(0.0), bw_(0.0), nBins_(0) {}

        double min_;
        double max_;
        double bw_;
        std::string label_;
        unsigned nBins_;

        template <typename Numeric, class Axis> friend class HistoND;
        friend class DualHistoAxis;

        inline unsigned overflowIndex(
            const double x, unsigned* binNumber) const
        {
            if (x < min_)
                return 0U;
            else if (x >= max_)
                return 2U;
            else
            {
                const unsigned bin = static_cast<unsigned>((x - min_)/bw_);
                *binNumber = bin >= nBins_ ? nBins_ - 1U : bin;
                return 1U;
            }
        }

        unsigned overflowIndexWeighted(double x, unsigned* binNumber,
                                       double *weight) const;
    };
}

#endif // NPSTAT_HISTOAXIS_HH_

