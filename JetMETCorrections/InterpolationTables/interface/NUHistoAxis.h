#ifndef NPSTAT_NUHISTOAXIS_HH_
#define NPSTAT_NUHISTOAXIS_HH_

/*!
// \file NUHistoAxis.h
//
// \brief Histogram axis with non-uniform bin spacing
//
// Author: I. Volobouev
//
// December 2011
*/

#include <vector>
#include <utility>

#include "Alignment/Geners/interface/ClassId.hh"
#include "JetMETCorrections/InterpolationTables/interface/Interval.h"

namespace npstat {
    template <typename Numeric, class Axis> class HistoND;
    class DualHistoAxis;

    /**
    // This class can be used to create histograms with non-uniform binning
    */
    class NUHistoAxis
    {
    public:
        /**
        // The number of bin edges provided must be at least 2. Edge
        // coordinates will be sorted internally in the increasing order.
        // The number of bins will be less by 1 than the number of edges.
        */
        NUHistoAxis(const std::vector<double>& binEdges,
                    const char* label = 0);

        //@{
        /** Examine axis propoerties */
        inline double min() const {return min_;}
        inline double max() const {return max_;}
        inline Interval<double> interval() const
            {return Interval<double>(min_, max_);}
        inline double length() const {return max_ - min_;}
        inline unsigned nBins() const {return nBins_;}
        inline double binWidth(const int binNum) const
            {return binEdges_.at(binNum+1) - binEdges_.at(binNum);}
        inline const std::string& label() const {return label_;}
        inline bool isUniform() const {return uniform_;}
        //@}

        /** Return the coordinate of the given bin left edge */
        inline double leftBinEdge(const int binNum) const
            {return binEdges_.at(binNum);}

        /** Return the coordinate of the given bin right edge */
        inline double rightBinEdge(const int binNum) const
            {return binEdges_.at(binNum + 1);}

        /** Return the coordinate of the given bin center */
        inline double binCenter(const int binNum) const
            {return 0.5*(binEdges_.at(binNum) + binEdges_.at(binNum + 1));}

        /** Return the coordinate interval occupied by the given bin */
        inline Interval<double> binInterval(const int binNum) const
            {return Interval<double>(binEdges_.at(binNum),
                                     binEdges_.at(binNum + 1));}

        /** Change the axis label */
        inline void setLabel(const char* newlabel)
            {label_ = newlabel ? newlabel : "";}

        /**
        // This method returns -1 for values below the lower limit and
        // "nBins()" for values equal to or above the upper limit
        */
        int binNumber(double x) const;

        /**
        // Floating point bin number given the coordinate. Useful for
        // interpolation methods and such.
        */
        double fltBinNumber(double x, bool mapLeftEdgeTo0=true) const;

        /**
        // This method returns the closest valid bin number
        // (above 0 and below nBins() )
        */
        unsigned closestValidBin(double x) const;

        bool operator==(const NUHistoAxis&) const;
        bool operator!=(const NUHistoAxis&) const;

        /** Comparison of axis coordinates within given tolerance */
        bool isClose(const NUHistoAxis&, double tol) const;

        //@{
        /** Method related to "geners" I/O */
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static inline const char* classname() {return "npstat::NUHistoAxis";}
        static inline unsigned version() {return 1;}
        static NUHistoAxis* read(const gs::ClassId& id, std::istream& in);

    private:
        inline NUHistoAxis() : min_(0.0), max_(0.0),
                               nBins_(0), uniform_(false) {}

        NUHistoAxis(unsigned nBins, double min, double max,
                    const char* label = 0);

        double min_;
        double max_;
        std::vector<double> binEdges_;
        std::string label_;
        unsigned nBins_;
        bool uniform_;

        template <typename Numeric, class Axis> friend class HistoND;
        friend class DualHistoAxis;

        inline unsigned overflowIndex(
            const double x, unsigned* binNum) const
        {
            if (x < min_)
                return 0U;
            else if (x >= max_)
                return 2U;
            else
            {
                *binNum = binNumber(x);
                return 1U;
            }
        }

    };
}

#endif // NPSTAT_NUHISTOAXIS_HH_

