#ifndef NPSTAT_DUALHISTOAXIS_HH_
#define NPSTAT_DUALHISTOAXIS_HH_

/*!
// \file DualHistoAxis.h
//
// \brief Represent both equidistant and non-uniform histogram axis binning
//
// Author: I. Volobouev
//
// July 2012
*/

#include "JetMETCorrections/InterpolationTables/interface/HistoAxis.h"
#include "JetMETCorrections/InterpolationTables/interface/NUHistoAxis.h"

namespace npstat {
    /**
    // Histogram axis which can be either uniform or non-uniform.
    // Will work a little bit slower than either HistoAxis or NUHistoAxis,
    // but can be used in place of either one of them.
    */
    class DualHistoAxis
    {
    public:
        // Constructors
        inline DualHistoAxis(const NUHistoAxis& a)
            : a_(a), u_(1U, 0.0, 1.0), uniform_(false) {}

        inline DualHistoAxis(const HistoAxis& u)
            : a_(dummy_vec()), u_(u), uniform_(true) {}

        inline DualHistoAxis(const std::vector<double>& binEdges,
                             const char* label = 0)
            : a_(binEdges, label), u_(1U, 0.0, 1.0), uniform_(false) {}

        inline DualHistoAxis(unsigned nBins, double min, double max,
                             const char* label = 0)
            : a_(dummy_vec()), u_(nBins, min, max, label), uniform_(true) {}

        // Inspectors
        inline bool isUniform() const {return uniform_;}

        inline double min() const
            {return uniform_ ? u_.min() : a_.min();}

        inline double max() const
            {return uniform_ ? u_.max() : a_.max();}

        inline Interval<double> interval() const
            {return uniform_ ? u_.interval() : a_.interval();}

        inline double length() const
            {return uniform_ ? u_.length() : a_.length();}

        inline unsigned nBins() const
            {return uniform_ ? u_.nBins() : a_.nBins();}

        inline double binWidth(const int binNum) const
            {return uniform_ ? u_.binWidth(binNum) : a_.binWidth(binNum);}

        inline const std::string& label() const
            {return uniform_ ? u_.label() : a_.label();}

        inline double binCenter(const int binNum) const
            {return uniform_ ? u_.binCenter(binNum) : a_.binCenter(binNum);}

        inline double leftBinEdge(const int binNum) const
            {return uniform_ ? u_.leftBinEdge(binNum) : a_.leftBinEdge(binNum);}

        inline double rightBinEdge(const int binNum) const
            {return uniform_ ? u_.rightBinEdge(binNum) : a_.rightBinEdge(binNum);}

        inline Interval<double> binInterval(const int binNum) const
            {return uniform_ ? u_.binInterval(binNum) : a_.binInterval(binNum);}

        //@{
        /**
        // Return a pointer to the underlying axis. This will be
        // a null pointer if the axis does not correspond to the
        // constructed type.
        */
        inline const NUHistoAxis* getNUHistoAxis() const
            {return uniform_ ? static_cast<const NUHistoAxis*>(0) : &a_;}

        inline const HistoAxis* getHistoAxis() const
            {return uniform_ ? &u_ : static_cast<const HistoAxis*>(0);}
        //@}

        /** Modify the axis label */
        inline void setLabel(const char* newlabel)
            {uniform_ ? u_.setLabel(newlabel) : a_.setLabel(newlabel);}

        /**
        // This method returns arbitrary integer bin number.
        // Possible output depends on whether the axis is uniform or not.
        */
        inline int binNumber(const double x) const
            {return uniform_ ? u_.binNumber(x) : a_.binNumber(x);}

        /**
        // Floating point bin number given the coordinate. Useful for
        // interpolation methods and such.
        */
        inline double fltBinNumber(const double x,
                                   const bool mapLeftEdgeTo0=true) const
        {
            return uniform_ ? u_.fltBinNumber(x, mapLeftEdgeTo0) :
                              a_.fltBinNumber(x, mapLeftEdgeTo0);
        }

        /** Return the closest valid bin number (above 0 and below nBins() ) */
        inline unsigned closestValidBin(const double x) const
            {return uniform_ ? u_.closestValidBin(x) : a_.closestValidBin(x);}

        inline bool operator==(const DualHistoAxis& r) const
            {return uniform_ == r.uniform_ && a_ == r.a_ && u_ == r.u_;}

        inline bool operator!=(const DualHistoAxis& r) const
            {return !(*this == r);}

        /** Comparison within given tolerance */
        inline bool isClose(const DualHistoAxis& r, const double tol) const
        {
            return uniform_ == r.uniform_ &&
                   a_.isClose(r.a_, tol) && 
                   u_.isClose(r.u_, tol);
        }

        //@{
        // Method related to "geners" I/O
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static inline const char* classname() {return "npstat::DualHistoAxis";}
        static inline unsigned version() {return 1;}
        static DualHistoAxis* read(const gs::ClassId& id, std::istream& in);

    private:
        inline DualHistoAxis()
            : a_(dummy_vec()), u_(1U, 0.0, 1.0), uniform_(true) {}

        NUHistoAxis a_;
        HistoAxis u_;
        bool uniform_;

        template <typename Numeric, class Axis> friend class HistoND;

        inline unsigned overflowIndex(
            const double x, unsigned* binNumber) const
        {
            return uniform_ ? u_.overflowIndex(x, binNumber) : 
                              a_.overflowIndex(x, binNumber);
        }

        inline static std::vector<double> dummy_vec()
        {
            std::vector<double> vec(2, 0.0);
            vec[1] = 1.0;
            return vec;
        }

    };
}

#endif // NPSTAT_DUALHISTOAXIS_HH_

