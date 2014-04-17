#ifndef NPSTAT_HISTOND_HH_
#define NPSTAT_HISTOND_HH_

/*! 
// \file HistoND.h
//
// \brief Arbitrary-dimensional histogram template
//
// Author: I. Volobouev
//
// July 2010
*/

#include "JetMETCorrections/InterpolationTables/interface/ArrayND.h"
#include "JetMETCorrections/InterpolationTables/interface/HistoAxis.h"

namespace npstat {
    /**
    // (Almost) arbitrary-dimensional histogram with binning determined
    // by the second template parameter (typically HistoAxis or NUHistoAxis).
    // The dimensionality must not exceed CHAR_BIT*sizeof(unsigned long)-1
    // which is normally 31/63 on 32/64-bit systems.
    //
    // The template parameter class (Numeric) must be such that it can be
    // used as the template parameter of ArrayND class. For a typical usage
    // pattern, Numeric should also support operator += between itself and
    // the weights with which the histogram is filled (see, however, the
    // description of the "dispatch" method which is not subject to
    // this recommendation).
    //
    // If the "fillC" method is used to accumulate the data then the weights
    // must support multiplication by a double, and then it must be possible
    // to use the "+=" operator to add such a product to Numeric.
    //
    // Note that there are no methods which would allow the user to examine
    // the bin contents of the histogram using bin numbers. This is
    // intentional: almost always such examinations are performed in a loop
    // over indices, and it is more efficient to grab a reference to the 
    // underlying array using the "binContents()" method and then examine
    // that array directly.
    */
    template <typename Numeric, class Axis=HistoAxis>
    class HistoND
    {
        template <typename Num2, class Axis2> friend class HistoND;

    public:
        typedef Numeric value_type;
        typedef Axis axis_type;

        enum RebinType {
            SAMPLE = 0,
            SUM,
            AVERAGE
        };

        /** Main constructor for arbitrary-dimensional histograms */
        explicit HistoND(const std::vector<Axis>& axes, const char* title=0,
                         const char* accumulatedDataLabel=0);

        /** Convenience constructor for 1-d histograms */
        explicit HistoND(const Axis& xAxis, const char* title=0,
                         const char* accumulatedDataLabel=0);

        /** Convenience constructor for 2-d histograms */
        HistoND(const Axis& xAxis, const Axis& yAxis,
                const char* title=0, const char* accumulatedDataLabel=0);

        /** Convenience constructor for 3-d histograms */
        HistoND(const Axis& xAxis, const Axis& yAxis, const Axis& zAxis,
                const char* title=0, const char* accumulatedDataLabel=0);

        /** Convenience constructor for 4-d histograms */
        HistoND(const Axis& xAxis, const Axis& yAxis,
                const Axis& zAxis, const Axis& tAxis,
                const char* title=0, const char* accumulatedDataLabel=0);

        /** Convenience constructor for 5-d histograms */
        HistoND(const Axis& xAxis, const Axis& yAxis,
                const Axis& zAxis, const Axis& tAxis, const Axis& vAxis,
                const char* title=0, const char* accumulatedDataLabel=0);

        /**
        // Simple constructor for uniformly binned histograms without
        // axis labels. Sequence size returned by the size() method of
        // both "shape" and "boundingBox" arguments must be the same.
        */
        HistoND(const ArrayShape& shape, const BoxND<double>& boundingBox,
                const char* title=0, const char* accumulatedDataLabel=0);

        /**
        // Converting constructor. The functor will be applied to all bins
        // of the argument histogram to fill the bins of the constructed
        // histogram. If the title and data label are not provided, they
        // will be cleared.
        */
        template <typename Num2, class Functor>
        HistoND(const HistoND<Num2,Axis>& h, const Functor& f,
                const char* title=0, const char* accumulatedDataLabel=0);

        /**
        // A slicing constructor. The new histogram will be created by
        // slicing another histogram. See the description of the slicing
        // constructor in the "ArrayND" class for the meaning of arguments
        // "indices" and "nIndices". The data of the newly created histogram
        // is cleared.
        */
        template <typename Num2>
        HistoND(const HistoND<Num2,Axis>& h, const unsigned *indices,
                unsigned nIndices, const char* title=0);

        /**
        // A constructor that inserts a new axis into a histogram
        // (as if the argument histogram was a slice of the new histogram).
        // The "newAxisNumber" argument specifies the number of the
        // new axis in the axis sequence of the constructed histogram.
        // If the "newAxisNumber" exceeds the number of axes of the
        // argument histogram, the new axis will become last. The data
        // of the newly created histogram is cleared.
        */
        template <typename Num2>
        HistoND(const HistoND<Num2,Axis>& h, const Axis& newAxis,
                unsigned newAxisNumber, const char* title=0);

        /**
        // Create a rebinned histogram with the same axis coverage.
        // Note that not all such operations will be meaningful if the
        // bin contents do not belong to one of the floating point types.
        // The "newBinCounts" argument specifies the new number of bins
        // along each axis. The length of this array (provided by the
        // "lenNewBinCounts" argument) should be equal to the input
        // histogram dimensionality.
        //
        // The "shifts" argument can be meaningfully specified with the
        // "rType" argument set to "SAMPLE". These shifts will be added
        // to the bin centers of the created histogram when the bin contents
        // are looked up in the input histogram. This can be useful in case
        // the bin center lookup without shifts would fall exactly on the
        // bin edge. Naturally, the length of the "shifts" array should be
        // equal to the input histogram dimensionality.
        */
        template <typename Num2>
        HistoND(const HistoND<Num2,Axis>& h, RebinType rType,
                const unsigned *newBinCounts, unsigned lenNewBinCounts, 
                const double* shifts=0, const char* title=0);

        /** Copy constructor */
        HistoND(const HistoND&);

        /** 
        // Assignment operator. Works even when the binning of the two
        // histograms is not compatible.
        */
        HistoND& operator=(const HistoND&);

        /** Histogram dimensionality */
        inline unsigned dim() const {return dim_;}

        /** Histogram title */
        inline const std::string& title() const {return title_;}

        /** Label associated with accumulated data */
        inline const std::string& accumulatedDataLabel() const
            {return accumulatedDataLabel_;}

        /** Retrive a reference to the array of bin contents */
        inline const ArrayND<Numeric>& binContents() const {return data_;}

        /** Retrive a reference to the array of overflows */
        inline const ArrayND<Numeric>& overflows() const {return overflow_;}

        /** Inspect histogram axes */
        inline const std::vector<Axis>& axes() const {return axes_;}

        /** Inspect a histogram axis for the given dimension */
        inline const Axis& axis(const unsigned i) const
            {return axes_.at(i);}

        /** Total number of bins */
        inline unsigned long nBins() const {return data_.length();}

        /** Total number of fills performed */
        inline unsigned long nFillsTotal() const {return fillCount_;}

        /** Total number of fills which fell inside the histogram range */
        inline unsigned long nFillsInRange() const
            {return fillCount_ - overCount_;}

        /** Total number of fills which fell outside the histogram range */
        inline unsigned long nFillsOver() const {return overCount_;}

        /** 
        // This method returns "true" if the method isUniform()
        // of each histogram axis returns "true" 
        */
        bool isUniformlyBinned() const;

        /** Modify the histogram title */
        inline void setTitle(const char* newtitle)
            {title_ = newtitle ? newtitle : ""; ++modCount_;}

        /** Modify the label associated with accumulated data */
        inline void setAccumulatedDataLabel(const char* newlabel)
            {accumulatedDataLabel_ = newlabel ? newlabel : ""; ++modCount_;}

        /** Modify the label for the histogram axis with the given number */
        inline void setAxisLabel(const unsigned axisNum, const char* newlabel)
            {axes_.at(axisNum).setLabel(newlabel); ++modCount_;}

        /**
        // This method returns width/area/volume/etc. of a single bin.
        // 1.0 is returned for a dimensionless histogram.
        */
        double binVolume(unsigned long binNumber=0) const;

        /**
        // Position of the bin center. Length of the "coords" array
        // (filled on return) should be equal to the dimensionality
        // of the histogram.
        */
        void binCenter(unsigned long binNumber,
                       double* coords, unsigned lenCoords) const;

        /**
        // Convenience function which fills out a vector of bin centers
        // in the same order as the linear order of binContents().
        // The class "Point" must have a subscript operator, default
        // constructor, copy constructor, and the size() method (use,
        // for example, std::array).
        */
        template <class Point>
        void allBinCenters(std::vector<Point>* centers) const;

        /** Bounding box for the given bin */
        void binBox(unsigned long binNumber, BoxND<double>* box) const;

        /** Bounding box for the whole histogram */
        BoxND<double> boundingBox() const;

        /**
        // Volume of the histogram bounding box (this direct call is faster
        // than calling boundingBox().volume() ). This function returns 1.0
        // for 0-dim histogram, axis interval length for 1-d histogram, etc.
        */
        double volume() const;

        /** Integral of the histogram */
        double integral() const;

        /** Clear the histogram contents (both bins and overflows) */
        void clear();

        /** This method clears the bin contents but not overflows */
        void clearBinContents();

        /** This method clears overflows but not the bin contents */
        void clearOverflows();

        /** Comparison for equality */
        bool operator==(const HistoND&) const;

        /** Logical negation of operator== */
        bool operator!=(const HistoND&) const;

        /** 
        // Check data for equality (both bin contents and overflows).
        // Do not compare axes, labels, fill counts, etc.
        */
        bool isSameData(const HistoND&) const;

        /**
        // Fill function for histograms of arbitrary dimensionality.
        // The length of the "coords" array should be equal to the
        // histogram dimensionality. The Numeric type must have the "+="
        // operator defined with the Num2 type on the right side.
        */
        template <typename Num2>
        void fill(const double* coords, unsigned coordLength,
                  const Num2& weight);

        //@{
        /**
        // Convenience "fill" method for histograms of corresponding
        // dimensionality
        */
        template <typename Num2>
        void fill(const Num2& weight);

        template <typename Num2>
        void fill(double x0, const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, double x3,
                  const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, double x3, double x4,
                  const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, double x3, double x4,
                  double x5, const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, double x3, double x4,
                  double x5, double x6, const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, double x3, double x4,
                  double x5, double x6, double x7, const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, double x3, double x4,
                  double x5, double x6, double x7, double x8,
                  const Num2& weight);

        template <typename Num2>
        void fill(double x0, double x1, double x2, double x3, double x4,
                  double x5, double x6, double x7, double x8, double x9,
                  const Num2& weight);
        //@}

        /**
        // Location-based dispatch method. The provided binary functor
        // will be called with the approprite histogram bin value as the
        // first argument and the weight as the second (functor return value
        // is ignored). This allows for a very general use of the histogram
        // binning functionality. For example, with a proper functor, the
        // histogram bins can be filled with pointers to an arbitrary class
        // (this is the only way to use classes which do not have default
        // constructors as bin contents) and the functor can be used to
        // dispatch class methods. Depending on the exact nature of the
        // functor, multiple things might be modified as the result of this
        // call: the bin value, the weight, and the functor internal state.
        */
        template <typename Num2, class Functor>
        void dispatch(const double* coords, unsigned coordLength,
                      Num2& weight, Functor& f);

        //@{
        /**
        // Convenience "dispatch" method for histograms of corresponding
        // dimensionality
        */
        template <typename Num2, class Functor>
        void dispatch(Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, Num2& weight,
                      Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, double x3,
                      Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, double x3, double x4,
                      Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, double x3, double x4,
                      double x5, Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, double x3, double x4,
                      double x5, double x6, Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, double x3, double x4,
                      double x5, double x6, double x7, Num2& weight,
                      Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, double x3, double x4,
                      double x5, double x6, double x7, double x8,
                      Num2& weight, Functor& f);

        template <typename Num2, class Functor>
        void dispatch(double x0, double x1, double x2, double x3, double x4,
                      double x5, double x6, double x7, double x8, double x9,
                      Num2& weight, Functor& f);
        //@}

        /**
        // The "examine" functions allow the user to access bin contents
        // when bins are addressed by their coordinates. Use "binContents()"
        // to access the data by bin numbers. Overflow bins will be accessed
        // if the given coordinates fall outside the histogram range.
        */
        const Numeric& examine(const double* coords,
                               unsigned coordLength) const;

        //@{
        /**
        // Convenience "examine" method for histograms of corresponding
        // dimensionality
        */
        const Numeric& examine() const;

        const Numeric& examine(double x0) const;

        const Numeric& examine(double x0, double x1) const;

        const Numeric& examine(double x0, double x1, double x2) const;

        const Numeric& examine(double x0, double x1, double x2,
                               double x3) const;

        const Numeric& examine(double x0, double x1, double x2, double x3,
                               double x4) const;

        const Numeric& examine(double x0, double x1, double x2, double x3,
                               double x4, double x5) const;

        const Numeric& examine(double x0, double x1, double x2, double x3,
                               double x4, double x5, double x6) const;

        const Numeric& examine(double x0, double x1, double x2, double x3,
                               double x4, double x5, double x6,
                               double x7) const;

        const Numeric& examine(double x0, double x1, double x2, double x3,
                               double x4, double x5, double x6, double x7,
                               double x8) const;

        const Numeric& examine(double x0, double x1, double x2, double x3,
                               double x4, double x5, double x6, double x7,
                               double x8, double x9) const;
        //@}

        /**
        // The "closestBin" functions are similar to the "examine" functions
        // but always return a valid bin and never overflow. This can be
        // useful for implementing lookup tables with constant extrapolation
        // outside of the histogram range.
        */
        const Numeric& closestBin(const double* coords,
                                  unsigned coordLength) const;

        //@{
        /**
        // Convenience "closestBin" method for histograms of corresponding
        // dimensionality
        */
        const Numeric& closestBin() const;

        const Numeric& closestBin(double x0) const;

        const Numeric& closestBin(double x0, double x1) const;

        const Numeric& closestBin(double x0, double x1, double x2) const;

        const Numeric& closestBin(double x0, double x1, double x2,
                                  double x3) const;

        const Numeric& closestBin(double x0, double x1, double x2, double x3,
                                  double x4) const;

        const Numeric& closestBin(double x0, double x1, double x2, double x3,
                                  double x4, double x5) const;

        const Numeric& closestBin(double x0, double x1, double x2, double x3,
                                  double x4, double x5, double x6) const;

        const Numeric& closestBin(double x0, double x1, double x2, double x3,
                                  double x4, double x5, double x6,
                                  double x7) const;

        const Numeric& closestBin(double x0, double x1, double x2, double x3,
                                  double x4, double x5, double x6, double x7,
                                  double x8) const;

        const Numeric& closestBin(double x0, double x1, double x2, double x3,
                                  double x4, double x5, double x6, double x7,
                                  double x8, double x9) const;
        //@}

        /**
        // The "fillC" functions are similar to the "fill" methods but
        // they preserve the centroid of the deposit. Note that, if the
        // histogram dimensionality is high, "fillC" works significantly
        // slower than the corresponding "fill". Also note that there
        // must be at least 2 bins in each dimension in order for this
        // function to work.
        //
        // A word of caution. What is added to the bins is the input weight
        // multiplied by another weight calculated using the bin proximity.
        // If the input weight is just 1 (which happens quite often in
        // practice), the product of the weights is normally less than 1.
        // If the histogram template parameter is one of the integer types,
        // operator += will convert this product to 0 before adding it to
        // the bin! Therefore, it is best to use "fillC" only with floating
        // point template parameters (float, double, etc).
        // 
        // Currently, the "fillC" methods work sensibly only in the case
        // the binning is uniform (i.e., the second template parameter is
        // HistoAxis rather than, let say, NUHistoAxis). They typically
        // will not even compile if the binning is not uniform.
        */
        template <typename Num2>
        void fillC(const double* coords, unsigned coordLength,
                   const Num2& weight);

        //@{
        /**
        // Convenience "fillC" method for histograms of corresponding
        // dimensionality
        */
        template <typename Num2>
        void fillC(const Num2& weight);

        template <typename Num2>
        void fillC(double x0, const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, double x2, const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, double x2, double x3,
                   const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, double x2, double x3, double x4,
                   const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, double x2, double x3, double x4,
                   double x5, const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, double x2, double x3, double x4,
                   double x5, double x6, const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, double x2, double x3, double x4,
                   double x5, double x6, double x7, const Num2& weight);

        template <typename Num2>
        void fillC(double x0, double x1, double x2, double x3, double x4,
                   double x5, double x6, double x7, double x8,
                   const Num2& weight);
        
        template <typename Num2>
        void fillC(double x0, double x1, double x2, double x3, double x4,
                   double x5, double x6, double x7, double x8, double x9,
                   const Num2& weight);
        //@}

        /**
        // Fill from another histogram. Compatibility of axis limits
        // will not be checked, but compatibility of array shapes will be.
        */
        template <typename Num2>
        HistoND& operator+=(const HistoND<Num2,Axis>& r);

        /**
        // Subtract contents of another histogram. Equivalent to multiplying
        // the contents of the other histogram by -1 and then adding them.
        // One of the consequences of this approach is that, for histograms
        // "a" and "b", the sequence of operations "a += b; a -= b;" does not
        // leave histogram "a" unchanged: although its bin contents will
        // remain the same (up to round-off errors), the fill counts will
        // increase by twice the fill counts of "b".
        */
        template <typename Num2>
        HistoND& operator-=(const HistoND<Num2,Axis>& r);

        //@{
        /** Method to set contents of individual bins (no bounds checking) */
        template <typename Num2>
        void setBin(const unsigned *index, unsigned indexLen, const Num2& v);

        template <typename Num2>
        void setBin(const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                    const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                    unsigned i4, const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                    unsigned i4, unsigned i5, const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                    unsigned i4, unsigned i5, unsigned i6, const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                    unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                    const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                    unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                    unsigned i8, const Num2& v);

        template <typename Num2>
        void setBin(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                    unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                    unsigned i8, unsigned i9, const Num2& v);

        template <typename Num2>
        inline void setLinearBin(const unsigned long index, const Num2& v)
            {data_.linearValue(index) = v; ++modCount_;}
        //@}

        //@{
        /** Method to set contents of individual bins with bounds checking */
        template <typename Num2>
        void setBinAt(const unsigned *index, unsigned indexLen, const Num2& v);

        template <typename Num2>
        void setBinAt(const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                      const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                      unsigned i4, const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                      unsigned i4, unsigned i5, const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                      unsigned i4, unsigned i5, unsigned i6, const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                      unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                      const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                      unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                      unsigned i8, const Num2& v);

        template <typename Num2>
        void setBinAt(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                      unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                      unsigned i8, unsigned i9, const Num2& v);

        template <typename Num2>
        inline void setLinearBinAt(const unsigned long index, const Num2& v)
            {data_.linearValueAt(index) = v; ++modCount_;}
        //@}

        /** This method sets all bin contents in one fell swoop */
        template <typename Num2>
        void setBinContents(const Num2* data, unsigned long dataLength,
                            bool clearOverflows=true);

        /** This method sets all overflows in one fell swoop */
        template <typename Num2>
        void setOverflows(const Num2* data, unsigned long dataLength);

        /**
        // Setting bin contents to some constant value.
        // The Num2 type should allow automatic conversion to Numeric.
        */
        template <typename Num2>
        inline void setBinsToConst(const Num2& value)
            {data_.constFill(value); ++modCount_;}

        /**
        // Setting overflows to some constant value.
        // The Num2 type should allow automatic conversion to Numeric.
        */
        template <typename Num2>
        inline void setOverflowsToConst(const Num2& value)
            {overflow_.constFill(value); ++modCount_;}

        /**
        // This member function instructs the histogram to recalculate
        // the number of fills from data. It may be useful to call this
        // function after "setBinContents" in case the contents are filled
        // from another histogram.
        */
        void recalculateNFillsFromData();

        //@{
        /**
        // This method is intended for data format conversion
        // programs only, not for typical histogramming use
        */
        inline void setNFillsTotal(const unsigned long i)
            {fillCount_ = i; ++modCount_;}
        inline void setNFillsOver(const unsigned long i)
            {overCount_ = i; ++modCount_;}
        //@}

        /** In-place multiplication by a scalar (scaling) */
        template <typename Num2>
        HistoND& operator*=(const Num2& r);

        /** In-place division by a scalar */
        template <typename Num2>
        HistoND& operator/=(const Num2& r);

        //@{
        /** Multiplication by a value which is different for every bin */
        template <typename Num2>
        void scaleBinContents(const Num2* data, unsigned long dataLength);

        template <typename Num2>
        void scaleOverflows(const Num2* data, unsigned long dataLength);
        //@}

        //@{
        /**
        // In-place addition of a scalar to all bins. Equivalent to calling
        // the "fill" function with the same weight once for every bin.
        */
        template <typename Num2>
        void addToBinContents(const Num2& weight);

        template <typename Num2>
        void addToOverflows(const Num2& weight);
        //@}

        //@{
        /**
        // In-place addition of an array. Equivalent to calling the "fill"
        // function once for every bin with the weight taken from the
        // corresponding array element.
        */
        template <typename Num2>
        void addToBinContents(const Num2* data, unsigned long dataLength);

        template <typename Num2>
        void addToOverflows(const Num2* data, unsigned long dataLength);
        //@}

        /**
        // Add contents of all bins inside the given box to the accumulator.
        // Note that Numeric type must support multiplication by a double
        // in order for this function to work (it calculates the overlap
        // fraction of each bin with the box and multiplies bin content
        // by that fraction for subsequent accumulation). The operation
        // Acc += Numeric must be defined.
        */
        template <typename Acc>
        void accumulateBinsInBox(const BoxND<double>& box, Acc* acc,
                                 bool calculateAverage = false) const;

        //@{
        /**
        // Code for projecting one histogram onto another. For now,
        // this is done for bin contents only, not for overflows.
        // The projection should be created in advance from this
        // histogram with the aid of the slicing constructor. The indices
        // used in that constructor should be provided here as well.
        //
        // Note that you might want to recalculate the number of fills
        // from data after performing all projections needed.
        */
        template <typename Num2, typename Num3>
        void addToProjection(HistoND<Num2,Axis>* projection,
                             AbsArrayProjector<Numeric,Num3>& projector,
                             const unsigned *projectedIndices,
                             unsigned nProjectedIndices) const;

        template <typename Num2, typename Num3>
        void addToProjection(HistoND<Num2,Axis>* projection,
                             AbsVisitor<Numeric,Num3>& projector,
                             const unsigned *projectedIndices,
                             unsigned nProjectedIndices) const;
        //@}

        /** Transpose the histogram axes and bin contents */
        HistoND transpose(unsigned axisNum1, unsigned axisNum2) const;

        /**
        // This method returns the number of modifications
        // performed on the histogram since its creation. This number
        // is always increasing during the lifetime of the histogram
        // object. Its main property is as follows: if the method
        // "getModCount" returns the same number twice, there should
        // be no changes in the histogram object (so that a drawing
        // program does not need to redraw the histogram image).
        //
        // This number is pure transient, it is not serialized and
        // does not participate in histogram comparisons for equality.
        */
        inline unsigned long getModCount() const {return modCount_;}

        /**
        // Indicate that the histogram contents have changed. Should
        // be used by any code which directly modifies histogram bins
        // (after using const_cast on the relevant reference).
        */
        inline void incrModCount() {++modCount_;}

        //@{
        /** Method related to "geners" I/O */
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static const char* classname();
        static inline unsigned version() {return 1;}
        static HistoND* read(const gs::ClassId& id, std::istream& in);

    private:
        HistoND();

        // Special constructor which speeds up the "transpose" operation.
        // Does not do full error checking (some of it is done in transpose).
        HistoND(const HistoND& r, unsigned ax1, unsigned ax2);

        template <typename Num2>
        void fillPreservingCentroid(const Num2& weight);

        template <typename Acc>
        void accumulateBinsLoop(unsigned level, const BoxND<double>& box,
                                unsigned* idx, Acc* accumulator,
                                double overlapFraction, long double* wsum) const;
        std::string title_;
        std::string accumulatedDataLabel_;
        ArrayND<Numeric> data_;
        ArrayND<Numeric> overflow_;
        std::vector<Axis> axes_;
        mutable std::vector<double> weightBuf_;
        mutable std::vector<unsigned> indexBuf_;
        unsigned long fillCount_;
        unsigned long overCount_;
        unsigned long modCount_;
        unsigned dim_;

    };

    /**
    // Reset negative histogram bins to zero and then divide histogram
    // bin contents by the histogram integral. If the "knownNonNegative"
    // argument is true, it will be assumed that there are no negative
    // bins, and their explicit reset is unnecessary.
    //
    // This function will throw npstat::NpstatRuntimeError in case the histogram
    // is empty after all negative bins are reset.
    //
    // This function is not a member of the HistoND class itself because
    // these operations do not necessarily make sense for all bin types.
    // Making such operation a member would make creation of HistoND
    // scripting API (e.g., for python) more difficult.
    */
    template <typename Histo>
    void convertHistoToDensity(Histo* histogram, bool knownNonNegative=false);

    /**
    // Generate a density scanning map for subsequent use with
    // the "DensityScanND" template. Naturally, only histograms
    // with uniform binning can be used here.
    */
    template <typename Histo>
    std::vector<LinearMapper1d> densityScanHistoMap(const Histo& histo);

    /**
    // Generate a density scanning map for subsequent use with the
    // "DensityScanND" template when a density is to be convolved with
    // the histogram data. Only histograms with uniform binning
    // can be used here.
    //
    // The "doubleDataRange" should be set "true" in case the data
    // will be mirrored (or just empty range added) to avoid circular
    // spilling after convolution.
    */
    template <typename Histo>
    std::vector<CircularMapper1d> convolutionHistoMap(const Histo& histo,
                                                      bool doubleDataRange);
}

#include <cassert>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"
#include <sstream>
#include <climits>
#include <algorithm>

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/binaryIO.hh"

namespace npstat {
    namespace Private {
        template <class Axis>
        ArrayShape makeHistoShape(const std::vector<Axis>& axes)
        {
            const unsigned n = axes.size();
            ArrayShape result;
            result.reserve(n);
            for (unsigned i=0; i<n; ++i)
                result.push_back(axes[i].nBins());
            return result;
        }

        template <class Axis>
        ArrayShape makeHistoShape(const Axis& xAxis)
        {
            ArrayShape result;
            result.reserve(1U);
            result.push_back(xAxis.nBins());
            return result;
        }

        template <class Axis>
        ArrayShape makeHistoShape(const Axis& xAxis, const Axis& yAxis)
        {
            ArrayShape result;
            result.reserve(2U);
            result.push_back(xAxis.nBins());
            result.push_back(yAxis.nBins());
            return result;
        }

        template <class Axis>
        ArrayShape makeHistoShape(const Axis& xAxis,
                                  const Axis& yAxis,
                                  const Axis& zAxis)
        {
            ArrayShape result;
            result.reserve(3U);
            result.push_back(xAxis.nBins());
            result.push_back(yAxis.nBins());
            result.push_back(zAxis.nBins());
            return result;
        }

        template <class Axis>
        ArrayShape makeHistoShape(const Axis& xAxis, const Axis& yAxis,
                                  const Axis& zAxis, const Axis& tAxis)
        {
            ArrayShape result;
            result.reserve(4U);
            result.push_back(xAxis.nBins());
            result.push_back(yAxis.nBins());
            result.push_back(zAxis.nBins());
            result.push_back(tAxis.nBins());
            return result;
        }

        template <class Axis>
        ArrayShape makeHistoShape(const Axis& xAxis, const Axis& yAxis,
                                  const Axis& zAxis, const Axis& tAxis,
                                  const Axis& vAxis)
        {
            ArrayShape result;
            result.reserve(5U);
            result.push_back(xAxis.nBins());
            result.push_back(yAxis.nBins());
            result.push_back(zAxis.nBins());
            result.push_back(tAxis.nBins());
            result.push_back(vAxis.nBins());
            return result;
        }

        template <class Axis>
        std::vector<Axis> rebinAxes(const std::vector<Axis>& axes,
                                    const unsigned *newBins,
                                    const unsigned lenNewBins)
        {
            const unsigned dim = axes.size();
            if (lenNewBins != dim) throw npstat::NpstatInvalidArgument(
                "In npstat::Private::rebinAxes: invalid length "
                "of the new bins array");
            assert(newBins);
            std::vector<Axis> newAxes;
            newAxes.reserve(dim);
            for (unsigned i=0; i<dim; ++i)
                newAxes.push_back(axes[i].rebin(newBins[i]));
            return newAxes;
        }

        template <class Axis>
        std::vector<Axis> axesOfASlice(const std::vector<Axis>& axes,
                                       const unsigned *fixedIndices,
                                       const unsigned nFixedIndices)
        {
            const unsigned dim = axes.size();
            std::vector<Axis> newAxes;
            if (nFixedIndices == 0U) throw npstat::NpstatInvalidArgument(
                "In npstat::Private::axesOfASlice: "
                "at least one fixed index must be specified");
            if (nFixedIndices > dim) throw npstat::NpstatInvalidArgument(
                "In npstat::Private::axesOfASlice: too many fixed indices");
            assert(fixedIndices);
            for (unsigned i=0; i<nFixedIndices; ++i)
               if (fixedIndices[i] >= dim) throw npstat::NpstatInvalidArgument(
                  "In npstat::Private::axesOfASlice: fixed index out of range");
            newAxes.reserve(dim - nFixedIndices);
            for (unsigned i=0; i<dim; ++i)
            {
                bool fixed = false;
                for (unsigned j=0; j<nFixedIndices; ++j)
                    if (fixedIndices[j] == i)
                    {
                        fixed = true;
                        break;
                    }
                if (!fixed)
                    newAxes.push_back(axes[i]);
            }
            if (newAxes.size() != dim - nFixedIndices)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::Private::axesOfASlice: duplicate fixed index");
            return newAxes;
        }

        template <class Axis>
        ArrayShape shapeOfASlice(const std::vector<Axis>& axes,
                                 const unsigned *fixedIndices,
                                 const unsigned nFixedIndices)
        {
            const unsigned dim = axes.size();
            if (nFixedIndices == 0U) throw npstat::NpstatInvalidArgument(
                "In npstat::Private::shapeOfASlice: "
                "at least one fixed index must be specified");
            if (nFixedIndices > dim) throw npstat::NpstatInvalidArgument(
                "In npstat::Private::shapeOfASlice: too many fixed indices");
            assert(fixedIndices);

            // Check that the fixed indices are within range
            for (unsigned j=0; j<nFixedIndices; ++j)
              if (fixedIndices[j] >= dim) throw npstat::NpstatInvalidArgument(
                "In npstat::Private::shapeOfASlice: fixed index out of range");

            // Build the shape for the slice
            ArrayShape sh;
            if (nFixedIndices < dim)
                sh.reserve(dim - nFixedIndices);
            for (unsigned i=0; i<dim; ++i)
            {
                bool fixed = false;
                for (unsigned j=0; j<nFixedIndices; ++j)
                    if (fixedIndices[j] == i)
                    {
                        fixed = true;
                        break;
                    }
                if (!fixed)
                    sh.push_back(axes[i].nBins());
            }
            if (sh.size() != dim - nFixedIndices)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::Private::shapeOfASlice: duplicate fixed index");
            return sh;
        }

        template <class Axis>
        std::vector<Axis> addAxis(const std::vector<Axis>& axes,
                                  const Axis& newAxis,
                                  const unsigned newAxisNumber)
        {
            const unsigned dim = axes.size();
            std::vector<Axis> newAxes;
            newAxes.reserve(dim + 1U);
            unsigned iadd = 0;
            for (unsigned i=0; i<dim; ++i)
            {
                if (newAxisNumber == i)
                    newAxes.push_back(newAxis);
                else
                    newAxes.push_back(axes[iadd++]);
            }
            if (iadd == dim)
                newAxes.push_back(newAxis);
            else
                newAxes.push_back(axes[iadd]);
            return newAxes;
        }

        template <class Axis>
        ArrayShape shapeWithExtraAxis(const std::vector<Axis>& axes,
                                      const Axis& newAxis,
                                      const unsigned newAxisNumber)
        {
            const unsigned dim = axes.size();
            ArrayShape result;
            result.reserve(dim + 1U);
            unsigned iadd = 0;
            for (unsigned i=0; i<dim; ++i)
            {
                if (newAxisNumber == i)
                    result.push_back(newAxis.nBins());
                else
                    result.push_back(axes[iadd++].nBins());
            }
            if (iadd == dim)
                result.push_back(newAxis.nBins());
            else
                result.push_back(axes[iadd].nBins());
            return result;
        }

        inline void h_badargs(const char* method)
        {
            std::ostringstream os;
            os << "In npstat::HistoND::" << method << ": number of arguments"
               << " is incompatible with histogram dimensionality";
            throw npstat::NpstatInvalidArgument(os.str());
        }
    }

    template <typename Numeric, class Axis>
    template <typename Acc>
    void HistoND<Numeric,Axis>::accumulateBinsLoop(
        const unsigned level, const BoxND<double>& box,
        unsigned* idx, Acc* accumulator, const double overlapFraction,
        long double* wsum) const
    {
        const Interval<double>& boxSide(box[level]);
        const Axis& axis(axes_[level]);
        const unsigned nbins = axis.nBins();
        const bool lastLevel = level == dim_ - 1U;
        for (unsigned i=0; i<nbins; ++i)
        {
            const double over = overlapFraction*
                axis.binInterval(i).overlapFraction(boxSide);
            if (over > 0.0)
            {
                idx[level] = i;
                if (lastLevel)
                {
                    *accumulator += over*data_.value(idx, dim_);
                    *wsum += over;
                }
                else
                    accumulateBinsLoop(level+1U, box, idx, accumulator,
                                       over, wsum);
            }
        }
    }

    template <typename Numeric, class Axis>
    template <typename Acc>
    void HistoND<Numeric,Axis>::accumulateBinsInBox(
        const BoxND<double>& box, Acc* accumulator,
        const bool calculateAverage) const
    {
        if (box.size() != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::accumulateBinsInBox: "
            "incompatible box dimensionality");
        assert(accumulator);
        if (dim_)
        {
            long double wsum = 0.0L;
            for (unsigned i=0; i<dim_; ++i)
                indexBuf_[i] = 0U;
            accumulateBinsLoop(0U, box, &indexBuf_[0], accumulator, 1.0, &wsum);
            if (calculateAverage && wsum > 0.0L)
                *accumulator *= static_cast<double>(1.0L/wsum);
        }
        else
            *accumulator += 1.0*data_();
    }

    template <typename Numeric, class Axis>
    inline void HistoND<Numeric,Axis>::clearBinContents()
    {
        data_.clear();
        fillCount_ = 0UL;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    inline void HistoND<Numeric,Axis>::clearOverflows()
    {
        overflow_.clear();
        overCount_ = 0UL;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    inline void HistoND<Numeric,Axis>::clear()
    {
        clearBinContents();
        clearOverflows();
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const std::vector<Axis>& axesIn,
                                   const char* title, const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(Private::makeHistoShape(axesIn)),
          overflow_(ArrayShape(axesIn.size(), 3U)),
          axes_(axesIn),
          weightBuf_(axesIn.size()),
          indexBuf_(2U*axesIn.size()),
          modCount_(0UL),
          dim_(axesIn.size())
    {
        if (dim_ >= CHAR_BIT*sizeof(unsigned long))
            throw npstat::NpstatInvalidArgument(
                "In npstat::HistoND constructor: requested histogram "
                "dimensionality is not supported (too large)");
        clear();
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const Axis& xAxis,
                                   const char* title, const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(Private::makeHistoShape(xAxis)),
          overflow_(ArrayShape(1U, 3U)),
          weightBuf_(1U),
          indexBuf_(2U*1U),
          modCount_(0UL),
          dim_(1U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        clear();
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const Axis& xAxis, const Axis& yAxis,
                                   const char* title, const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(Private::makeHistoShape(xAxis, yAxis)),
          overflow_(ArrayShape(2U, 3U)),
          weightBuf_(2U),
          indexBuf_(2U*2U),
          modCount_(0UL),
          dim_(2U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);
        clear();
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const Axis& xAxis, const Axis& yAxis,
                                   const Axis& zAxis, const char* title,
                                   const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(Private::makeHistoShape(xAxis, yAxis, zAxis)),
          overflow_(ArrayShape(3U, 3U)),
          weightBuf_(3U),
          indexBuf_(2U*3U),
          modCount_(0UL),
          dim_(3U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);
        axes_.push_back(zAxis);
        clear();
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const Axis& xAxis, const Axis& yAxis,
                                   const Axis& zAxis, const Axis& tAxis,
                                   const char* title, const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(Private::makeHistoShape(xAxis, yAxis, zAxis, tAxis)),
          overflow_(ArrayShape(4U, 3U)),
          weightBuf_(4U),
          indexBuf_(2U*4U),
          modCount_(0UL),
          dim_(4U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);
        axes_.push_back(zAxis);
        axes_.push_back(tAxis);
        clear();
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const Axis& xAxis, const Axis& yAxis,
                                   const Axis& zAxis, const Axis& tAxis,
                                   const Axis& vAxis,
                                   const char* title, const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(Private::makeHistoShape(xAxis, yAxis, zAxis, tAxis, vAxis)),
          overflow_(ArrayShape(5U, 3U)),
          weightBuf_(5U),
          indexBuf_(2U*5U),
          modCount_(0UL),
          dim_(5U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);
        axes_.push_back(zAxis);
        axes_.push_back(tAxis);
        axes_.push_back(vAxis);
        clear();
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const ArrayShape& shape,
                                   const BoxND<double>& boundingBox,
                                   const char* title, const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(shape),
          overflow_(ArrayShape(shape.size(), 3U)),
          weightBuf_(shape.size()),
          indexBuf_(2U*shape.size()),
          modCount_(0UL),
          dim_(shape.size())
    {
        if (boundingBox.size() != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND constructor: "
            "incompatible bounding box dimensionality");
        if (dim_ >= CHAR_BIT*sizeof(unsigned long))
            throw npstat::NpstatInvalidArgument(
                "In npstat::HistoND constructor: requested histogram "
                "dimensionality is not supported (too large)");
        axes_.reserve(dim_);
        for (unsigned i=0; i<dim_; ++i)
            axes_.push_back(Axis(shape[i],
                                 boundingBox[i].min(),
                                 boundingBox[i].max()));
        clear();
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    HistoND<Numeric,Axis>::HistoND(
        const HistoND<Num2,Axis>& r, const Functor& f,
        const char* title, const char* label)
        : title_(title ? title : ""),
          accumulatedDataLabel_(label ? label : ""),
          data_(r.data_, f),
          overflow_(r.overflow_, f),
          axes_(r.axes_),
          weightBuf_(r.dim_),
          indexBuf_(2U*r.dim_),
          fillCount_(r.fillCount_),
          overCount_(r.overCount_),
          modCount_(0UL),
          dim_(r.dim_)
    {
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    HistoND<Numeric,Axis>::HistoND(
        const HistoND<Num2,Axis>& h, const unsigned *indices,
        const unsigned nIndices, const char* title)
        : title_(title ? title : ""),
          accumulatedDataLabel_(h.accumulatedDataLabel_),
          data_(Private::shapeOfASlice(h.axes_, indices, nIndices)),
          overflow_(ArrayShape(data_.rank(), 3U)),
          axes_(Private::axesOfASlice(h.axes_, indices, nIndices)),
          weightBuf_(data_.rank()),
          indexBuf_(2U*data_.rank()),
          modCount_(0UL),
          dim_(data_.rank())
    {
        clear();
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    HistoND<Numeric,Axis>::HistoND(
        const HistoND<Num2,Axis>& h, const Axis& newAxis,
        const unsigned newAxisNumber, const char* title)
        : title_(title ? title : ""),
          accumulatedDataLabel_(h.accumulatedDataLabel_),
          data_(Private::shapeWithExtraAxis(h.axes_, newAxis, newAxisNumber)),
          overflow_(data_.rank(), 3U),
          axes_(Private::addAxis(h.axes_, newAxis, newAxisNumber)),
          weightBuf_(data_.rank()),
          indexBuf_(2U*data_.rank()),
          modCount_(0UL),
          dim_(data_.rank())
    {
        if (dim_ >= CHAR_BIT*sizeof(unsigned long))
            throw npstat::NpstatInvalidArgument(
                "In npstat::HistoND constructor: requested histogram "
                "dimensionality is not supported (too large)");
        clear();
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    HistoND<Numeric,Axis>::HistoND(
        const HistoND<Num2,Axis>& h, const RebinType rType,
        const unsigned *newBinCounts, const unsigned lenNewBinCounts,
        const double* shifts, const char* title)
        : title_(title ? title : h.title_.c_str()),
          accumulatedDataLabel_(h.accumulatedDataLabel_),
          data_(newBinCounts, lenNewBinCounts),
          overflow_(h.overflow_),
          axes_(Private::rebinAxes(h.axes_, newBinCounts, lenNewBinCounts)),
          weightBuf_(h.dim_),
          indexBuf_(2U*h.dim_),
          fillCount_(h.fillCount_),
          overCount_(h.overCount_),
          modCount_(0UL),
          dim_(h.dim_)
    {
        const unsigned long newBins = data_.length();
        const Axis* ax = &axes_[0];
        unsigned* ubuf = &indexBuf_[0];

        // Fill out the bins of the new histogram
        if (rType == SAMPLE)
        {
            double* buf = &weightBuf_[0];
            for (unsigned long ibin=0; ibin<newBins; ++ibin)
            {
                data_.convertLinearIndex(ibin, ubuf, dim_);
                if (shifts)
                    for (unsigned i=0; i<dim_; ++i)
                        buf[i] = ax[i].binCenter(ubuf[i]) + shifts[i];
                else
                    for (unsigned i=0; i<dim_; ++i)
                        buf[i] = ax[i].binCenter(ubuf[i]);
                data_.linearValue(ibin) = h.examine(buf, dim_);
            }
        }
        else
        {
            const Numeric zero = Numeric();
            BoxND<double> binLimits(dim_);
            for (unsigned long ibin=0; ibin<newBins; ++ibin)
            {
                data_.convertLinearIndex(ibin, ubuf, dim_);
                for (unsigned i=0; i<dim_; ++i)
                    binLimits[i] = ax[i].binInterval(ubuf[i]);
                Numeric& thisBin(data_.linearValue(ibin));
                thisBin = zero;
                h.accumulateBinsInBox(binLimits, &thisBin, rType == AVERAGE);
            }
        }
    }

    template <typename Numeric, class Axis>
    bool HistoND<Numeric,Axis>::isUniformlyBinned() const
    {
        for (unsigned i=0; i<dim_; ++i)
            if (!axes_[i].isUniform())
                return false;
        return true;
    }

    template <typename Numeric, class Axis>
    double HistoND<Numeric,Axis>::integral() const
    {
        typedef typename PreciseType<Numeric>::type Precise;

        if (dim_ == 0U)
            return 0.0;
        if (isUniformlyBinned())
        {
            Precise sum = data_.template sum<Precise>();
            return static_cast<double>(sum)*binVolume();
        }
        else
        {
            Precise sum = Precise();
            const Numeric* data = data_.data();
            const unsigned long len = data_.length();
            for (unsigned long i=0; i<len; ++i)
                sum += data[i]*binVolume(i);
            return static_cast<double>(sum);
        }
    }

    template <typename Numeric, class Axis>
    BoxND<double> HistoND<Numeric,Axis>::boundingBox() const
    {
        BoxND<double> box;
        if (dim_)
        {
            box.reserve(dim_);
            const Axis* ax = &axes_[0];
            for (unsigned i=0; i<dim_; ++i)
                box.push_back(ax[i].interval());
        }
        return box;
    }

    template <typename Numeric, class Axis>
    void HistoND<Numeric,Axis>::binCenter(
        const unsigned long binNumber,
        double* coords, const unsigned lenCoords) const
    {
        if (dim_ != lenCoords) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::binCenter: "
            "incompatible input point dimensionality");
        if (dim_)
        {
            assert(coords);
            data_.convertLinearIndex(binNumber, &indexBuf_[0], dim_);
            const Axis* ax = &axes_[0];
            for (unsigned i=0; i<dim_; ++i)
                coords[i] = ax[i].binCenter(indexBuf_[i]);
        }
    }

    template <typename Numeric, class Axis>
    template <class Point>
    void HistoND<Numeric,Axis>::allBinCenters(
        std::vector<Point>* centers) const
    {
        assert(centers);
        centers->clear();
        const unsigned long len = data_.length();
        centers->reserve(len);
        unsigned* ibuf = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        Point center;
        if (center.size() < dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::allBinCenters: "
            "incompatible point dimensionality (too small)");
        typename Point::value_type* cdat = &center[0];

        for (unsigned long i=0; i<len; ++i)
        {
            data_.convertLinearIndex(i, ibuf, dim_);
            for (unsigned idim=0; idim<dim_; ++idim)
                cdat[idim] = ax[idim].binCenter(ibuf[idim]);
            centers->push_back(center);
        }
    }

    template <typename Numeric, class Axis>
    void HistoND<Numeric,Axis>::binBox(const unsigned long binNumber,
                                       BoxND<double>* box) const
    {
        assert(box);
        box->clear();
        if (dim_)
        {
            box->reserve(dim_);
            data_.convertLinearIndex(binNumber, &indexBuf_[0], dim_);
            const Axis* ax = &axes_[0];
            for (unsigned i=0; i<dim_; ++i)
                box->push_back(ax[i].binInterval(indexBuf_[i]));
        }
    }

    template <typename Numeric, class Axis>
    inline bool HistoND<Numeric,Axis>::isSameData(const HistoND& r) const
    {
        return dim_ == r.dim_ &&
               overflow_ == r.overflow_ &&
               data_ == r.data_;
    }

    template <typename Numeric, class Axis>
    inline bool HistoND<Numeric,Axis>::operator==(const HistoND& r) const
    {
        return dim_ == r.dim_ &&
               fillCount_ == r.fillCount_ &&
               overCount_ == r.overCount_ &&
               title_ == r.title_ &&
               accumulatedDataLabel_ == r.accumulatedDataLabel_ &&
               axes_ == r.axes_ &&
               overflow_ == r.overflow_ &&
               data_ == r.data_;
    }

    template <typename Numeric, class Axis>
    inline bool HistoND<Numeric,Axis>::operator!=(const HistoND& r) const
    {
        return !(*this == r);
    }

    template <typename Numeric, class Axis>
    double HistoND<Numeric,Axis>::binVolume(
        const unsigned long binNumber) const
    {
        double v = 1.0;
        if (dim_)
        {
            data_.convertLinearIndex(binNumber, &indexBuf_[0], dim_);
            const Axis* ax = &axes_[0];
            for (unsigned i=0; i<dim_; ++i)
                v *= ax[i].binWidth(indexBuf_[i]);
        }
        return v;
    }

    template <typename Numeric, class Axis>
    double HistoND<Numeric,Axis>::volume() const
    {
        double v = 1.0;
        if (dim_)
        {
            const Axis* ax = &axes_[0];
            for (unsigned i=0; i<dim_; ++i)
                v *= (ax[i].max() - ax[i].min());
        }
        return v;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(
        const double* coords, const unsigned coordLength, const Num2& w)
    {
        if (coordLength != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::fill: "
            "incompatible input point dimensionality");
        if (coordLength)
        {
            assert(coords);
            unsigned* idx = &indexBuf_[0];
            unsigned* over = idx + dim_;
            const Axis* ax = &axes_[0];
            unsigned overflown = 0U;
            for (unsigned i=0; i<dim_; ++i)
            {
                over[i] = ax[i].overflowIndex(coords[i], idx + i);
                overflown |= (over[i] - 1U);
            }
            if (overflown)
            {
                overflow_.value(over, dim_) += w;
                ++overCount_;
            }
            else
                data_.value(idx, dim_) += w;
        }
        else
            data_() += w;
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(
        const double* coords, const unsigned coordLength, Num2& w, Functor& f)
    {
        if (coordLength != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::dispatch: "
            "incompatible input point dimensionality");
        if (coordLength)
        {
            assert(coords);
            unsigned* idx = &indexBuf_[0];
            unsigned* over = idx + dim_;
            const Axis* ax = &axes_[0];
            unsigned overflown = 0U;
            for (unsigned i=0; i<dim_; ++i)
            {
                over[i] = ax[i].overflowIndex(coords[i], idx + i);
                overflown |= (over[i] - 1U);
            }
            if (overflown)
                f(overflow_.value(over, dim_), w);
            else
                f(data_.value(idx, dim_), w);
        }
        else
            f(data_(), w);
         ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(
        const double* coords, const unsigned coordLength) const
    {
        if (coordLength != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::examine: "
            "incompatible input point dimensionality");
        if (coordLength)
        {
            assert(coords);
            unsigned* idx = &indexBuf_[0];
            unsigned* over = idx + dim_;
            const Axis* ax = &axes_[0];
            unsigned overflown = 0U;
            for (unsigned i=0; i<dim_; ++i)
            {
                over[i] = ax[i].overflowIndex(coords[i], idx + i);
                overflown |= (over[i] - 1U);
            }
            if (overflown)
                return overflow_.value(over, dim_);
            else
                return data_.value(idx, dim_);
        }
        else
            return data_();
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(
        const double* coords, const unsigned coordLength) const
    {
        if (coordLength != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::closestBin: "
            "incompatible input point dimensionality");
        if (coordLength)
        {
            assert(coords);
            unsigned* idx = &indexBuf_[0];
            const Axis* ax = &axes_[0];
            for (unsigned i=0; i<dim_; ++i)
                idx[i] = ax[i].closestValidBin(coords[i]);
            return data_.value(idx, dim_);
        }
        else
            return data_();
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillPreservingCentroid(const Num2& value)
    {
        const double* weights = &weightBuf_[0];
        const unsigned* cell = &indexBuf_[0];
        const unsigned long* strides = data_.strides();
        const unsigned long maxcycle = 1UL << dim_;
        for (unsigned long icycle=0; icycle<maxcycle; ++icycle)
        {
            double w = 1.0;
            unsigned long icell = 0UL;
            for (unsigned i=0; i<dim_; ++i)
            {
                if (icycle & (1UL << i))
                {
                    w *= (1.0 - weights[i]);
                    icell += strides[i]*(cell[i] + 1U);
                }
                else
                {
                    w *= weights[i];
                    icell += strides[i]*cell[i];
                }
            }
            data_.linearValue(icell) += (value * w);
        }
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(
        const double* coords, const unsigned coordLength, const Num2& w)
    {
        if (coordLength != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::fillC: "
            "incompatible input point dimensionality");
        if (coordLength)
        {
            assert(coords);
            double* wg = &weightBuf_[0];
            unsigned* idx = &indexBuf_[0];
            unsigned* over = idx + dim_;
            const Axis* ax = &axes_[0];
            unsigned overflown = 0U;
            for (unsigned i=0; i<dim_; ++i)
            {
                over[i] = ax[i].overflowIndexWeighted(coords[i], idx+i, wg+i);
                overflown |= (over[i] - 1U);
            }
            if (overflown)
            {
                overflow_.value(over, dim_) += w;
                ++overCount_;
            }
            else
                fillPreservingCentroid(w);
        }
        else
            data_() += w;
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::fill(const Num2& w)
    {
        if (dim_) Private::h_badargs("fill");
        data_() += w;
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    inline void HistoND<Numeric,Axis>::dispatch(Num2& w, Functor& f)
    {
        if (dim_) Private::h_badargs("dispatch");
        f(data_(), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::fillC(const Num2& w)
    {
        if (dim_) Private::h_badargs("fillC");
        data_() += w;
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    inline const Numeric& HistoND<Numeric,Axis>::examine() const
    {
        if (dim_) Private::h_badargs("examine");
        return data_();
    }

    template <typename Numeric, class Axis>
    inline const Numeric& HistoND<Numeric,Axis>::closestBin() const
    {
        if (dim_) Private::h_badargs("closestBin");
        return data_();
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const Num2& w)
    {
        if (dim_ != 1U) Private::h_badargs("fill");
        unsigned i0 = 0;
        const unsigned ov0 = axes_[0].overflowIndex(x0, &i0);
        if (ov0 == 1U)
            data_(i0) += w;
        else
        {
            overflow_(ov0) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, Num2& w, Functor& f)
    {
        if (dim_ != 1U) Private::h_badargs("dispatch");
        unsigned i0 = 0;
        const unsigned ov0 = axes_[0].overflowIndex(x0, &i0);
        if (ov0 == 1U)
            f(data_(i0), w);
        else
            f(overflow_(ov0), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const Num2& w)
    {
        if (dim_ != 1U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const unsigned ov0 = axes_[0].overflowIndexWeighted(x0, idx, wg);
        if (ov0 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(ov0) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    inline const Numeric& HistoND<Numeric,Axis>::examine(const double x0) const
    {
        if (dim_ != 1U) Private::h_badargs("examine");
        unsigned i0 = 0;
        const unsigned ov0 = axes_[0].overflowIndex(x0, &i0);
        if (ov0 == 1U)
            return data_(i0);
        else
            return overflow_(ov0);
    }

    template <typename Numeric, class Axis>
    inline const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0) const
    {
        if (dim_ != 1U) Private::h_badargs("closestBin");
        const unsigned i0 = axes_[0].closestValidBin(x0);
        return data_(i0);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const Num2& w)
    {
        if (dim_ != 2U) Private::h_badargs("fill");
        unsigned i0 = 0, i1 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        if (o0 == 1U && o1 == 1U)
            data_(i0, i1) += w;
        else
        {
            overflow_(o0, o1) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         Num2& w, Functor& f)
    {
        if (dim_ != 2U) Private::h_badargs("dispatch");
        unsigned i0 = 0, i1 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        if (o0 == 1U && o1 == 1U)
            f(data_(i0, i1), w);
        else
            f(overflow_(o0, o1), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const Num2& w)
    {
        if (dim_ != 2U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        if (o0 == 1U && o1 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(const double x0,
                                                  const double x1) const
    {
        if (dim_ != 2U) Private::h_badargs("examine");
        unsigned i0 = 0, i1 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        if (o0 == 1U && o1 == 1U)
            return data_(i0, i1);
        else
            return overflow_(o0, o1);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1) const
    {
        if (dim_ != 2U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        return data_(i0, i1);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const Num2& w)
    {
        if (dim_ != 3U) Private::h_badargs("fill");
        unsigned i0 = 0, i1 = 0, i2 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        if (o0 == 1U && o1 == 1U && o2 == 1U)
            data_(i0, i1, i2) += w;
        else
        {
            overflow_(o0, o1, o2) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, Num2& w, Functor& f)
    {
        if (dim_ != 3U) Private::h_badargs("dispatch");
        unsigned i0 = 0, i1 = 0, i2 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        if (o0 == 1U && o1 == 1U && o2 == 1U)
            f(data_(i0, i1, i2), w);
        else
            f(overflow_(o0, o1, o2), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                 const double x2, const Num2& w)
    {
        if (dim_ != 3U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        if (o0 == 1U && o1 == 1U && o2 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(const double x0,
                                                  const double x1,
                                                  const double x2) const
    {
        if (dim_ != 3U) Private::h_badargs("examine");
        unsigned i0 = 0, i1 = 0, i2 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        if (o0 == 1U && o1 == 1U && o2 == 1U)
            return data_(i0, i1, i2);
        else
            return overflow_(o0, o1, o2);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2) const
    {
        if (dim_ != 3U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        return data_(i0, i1, i2);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const double x3,
                                     const Num2& w)
    {
        if (dim_ != 4U) Private::h_badargs("fill");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U)
            data_(i0, i1, i2, i3) += w;
        else
        {
            overflow_(o0, o1, o2, o3) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, const double x3,
                                         Num2& w, Functor& f)
    {
        if (dim_ != 4U) Private::h_badargs("dispatch");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U)
            f(data_(i0, i1, i2, i3), w);
        else
            f(overflow_(o0, o1, o2, o3), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const double x2, const double x3,
                                      const Num2& w)
    {
        if (dim_ != 4U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        const unsigned o3 = ax[3].overflowIndexWeighted(x3, idx+3, wg+3);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2, o3) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(const double x0,
                                                  const double x1,
                                                  const double x2,
                                                  const double x3) const
    {
        if (dim_ != 4U) Private::h_badargs("examine");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U)
            return data_(i0, i1, i2, i3);
        else
            return overflow_(o0, o1, o2, o3);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2,
                                                     const double x3) const
    {
        if (dim_ != 4U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        const unsigned i3 = ax[3].closestValidBin(x3);
        return data_(i0, i1, i2, i3);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const double x3,
                                     const double x4, const Num2& w)
    {
        if (dim_ != 5U) Private::h_badargs("fill");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U)
            data_(i0, i1, i2, i3, i4) += w;
        else
        {
            overflow_(o0, o1, o2, o3, o4) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, const double x3,
                                         const double x4, Num2& w, Functor& f)
    {
        if (dim_ != 5U) Private::h_badargs("dispatch");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U)
            f(data_(i0, i1, i2, i3, i4), w);
        else
            f(overflow_(o0, o1, o2, o3, o4), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const double x2, const double x3,
                                      const double x4, const Num2& w)
    {
        if (dim_ != 5U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        const unsigned o3 = ax[3].overflowIndexWeighted(x3, idx+3, wg+3);
        const unsigned o4 = ax[4].overflowIndexWeighted(x4, idx+4, wg+4);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2, o3, o4) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(
        const double x0, const double x1,
        const double x2, const double x3,
        const double x4) const
    {
        if (dim_ != 5U) Private::h_badargs("examine");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U)
            return data_(i0, i1, i2, i3, i4);
        else
            return overflow_(o0, o1, o2, o3, o4);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2,
                                                     const double x3,
                                                     const double x4) const
    {
        if (dim_ != 5U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        const unsigned i3 = ax[3].closestValidBin(x3);
        const unsigned i4 = ax[4].closestValidBin(x4);
        return data_(i0, i1, i2, i3, i4);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const double x3,
                                     const double x4, const double x5,
                                     const Num2& w)
    {
        if (dim_ != 6U) Private::h_badargs("fill");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U)
            data_(i0, i1, i2, i3, i4, i5) += w;
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }
    
    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, const double x3,
                                         const double x4, const double x5,
                                         Num2& w, Functor& f)
    {
        if (dim_ != 6U) Private::h_badargs("dispatch");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U)
            f(data_(i0, i1, i2, i3, i4, i5), w);
        else
            f(overflow_(o0, o1, o2, o3, o4, o5), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const double x2, const double x3,
                                      const double x4, const double x5,
                                      const Num2& w)
    {
        if (dim_ != 6U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        const unsigned o3 = ax[3].overflowIndexWeighted(x3, idx+3, wg+3);
        const unsigned o4 = ax[4].overflowIndexWeighted(x4, idx+4, wg+4);
        const unsigned o5 = ax[5].overflowIndexWeighted(x5, idx+5, wg+5);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }
    
    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(const double x0,
                                                  const double x1,
                                                  const double x2,
                                                  const double x3,
                                                  const double x4,
                                                  const double x5) const
    {
        if (dim_ != 6U) Private::h_badargs("examine");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U)
            return data_(i0, i1, i2, i3, i4, i5);
        else
            return overflow_(o0, o1, o2, o3, o4, o5);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2,
                                                     const double x3,
                                                     const double x4,
                                                     const double x5) const
    {
        if (dim_ != 6U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        const unsigned i3 = ax[3].closestValidBin(x3);
        const unsigned i4 = ax[4].closestValidBin(x4);
        const unsigned i5 = ax[5].closestValidBin(x5);
        return data_(i0, i1, i2, i3, i4, i5);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const double x3,
                                     const double x4, const double x5,
                                     const double x6, const Num2& w)
    {
        if (dim_ != 7U) Private::h_badargs("fill");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U && o6 == 1U)
            data_(i0, i1, i2, i3, i4, i5, i6) += w;
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, const double x3,
                                         const double x4, const double x5,
                                         const double x6, Num2& w, Functor& f)
    {
        if (dim_ != 7U) Private::h_badargs("dispatch");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U && o6 == 1U)
            f(data_(i0, i1, i2, i3, i4, i5, i6), w);
        else
            f(overflow_(o0, o1, o2, o3, o4, o5, o6), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const double x2, const double x3,
                                      const double x4, const double x5,
                                      const double x6, const Num2& w)
    {
        if (dim_ != 7U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        const unsigned o3 = ax[3].overflowIndexWeighted(x3, idx+3, wg+3);
        const unsigned o4 = ax[4].overflowIndexWeighted(x4, idx+4, wg+4);
        const unsigned o5 = ax[5].overflowIndexWeighted(x5, idx+5, wg+5);
        const unsigned o6 = ax[6].overflowIndexWeighted(x6, idx+6, wg+6);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U && o6 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(
        const double x0, const double x1,
        const double x2, const double x3,
        const double x4, const double x5,
        const double x6) const
    {
        if (dim_ != 7U) Private::h_badargs("examine");
        unsigned i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        if (o0 == 1U && o1 == 1U && o2 == 1U &&
            o3 == 1U && o4 == 1U && o5 == 1U && o6 == 1U)
            return data_(i0, i1, i2, i3, i4, i5, i6);
        else
            return overflow_(o0, o1, o2, o3, o4, o5, o6);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2,
                                                     const double x3,
                                                     const double x4,
                                                     const double x5,
                                                     const double x6) const
    {
        if (dim_ != 7U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        const unsigned i3 = ax[3].closestValidBin(x3);
        const unsigned i4 = ax[4].closestValidBin(x4);
        const unsigned i5 = ax[5].closestValidBin(x5);
        const unsigned i6 = ax[6].closestValidBin(x6);
        return data_(i0, i1, i2, i3, i4, i5, i6);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const double x3,
                                     const double x4, const double x5,
                                     const double x6, const double x7,
                                     const Num2& w)
    {
        if (dim_ != 8U) Private::h_badargs("fill");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U &&
            o4 == 1U && o5 == 1U && o6 == 1U && o7 == 1U)
            data_(i0, i1, i2, i3, i4, i5, i6, i7) += w;
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6, o7) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, const double x3,
                                         const double x4, const double x5,
                                         const double x6, const double x7,
                                         Num2& w, Functor& f)
    {
        if (dim_ != 8U) Private::h_badargs("dispatch");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U &&
            o4 == 1U && o5 == 1U && o6 == 1U && o7 == 1U)
            f(data_(i0, i1, i2, i3, i4, i5, i6, i7), w);
        else
            f(overflow_(o0, o1, o2, o3, o4, o5, o6, o7), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const double x2, const double x3,
                                      const double x4, const double x5,
                                      const double x6, const double x7,
                                      const Num2& w)
    {
        if (dim_ != 8U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        const unsigned o3 = ax[3].overflowIndexWeighted(x3, idx+3, wg+3);
        const unsigned o4 = ax[4].overflowIndexWeighted(x4, idx+4, wg+4);
        const unsigned o5 = ax[5].overflowIndexWeighted(x5, idx+5, wg+5);
        const unsigned o6 = ax[6].overflowIndexWeighted(x6, idx+6, wg+6);
        const unsigned o7 = ax[7].overflowIndexWeighted(x7, idx+7, wg+7);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U &&
            o4 == 1U && o5 == 1U && o6 == 1U && o7 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6, o7) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(
        const double x0, const double x1,
        const double x2, const double x3,
        const double x4, const double x5,
        const double x6,
        const double x7) const
    {
        if (dim_ != 8U) Private::h_badargs("examine");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U &&
            o4 == 1U && o5 == 1U && o6 == 1U && o7 == 1U)
            return data_(i0, i1, i2, i3, i4, i5, i6, i7);
        else
            return overflow_(o0, o1, o2, o3, o4, o5, o6, o7);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2,
                                                     const double x3,
                                                     const double x4,
                                                     const double x5,
                                                     const double x6,
                                                     const double x7) const
    {
        if (dim_ != 8U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        const unsigned i3 = ax[3].closestValidBin(x3);
        const unsigned i4 = ax[4].closestValidBin(x4);
        const unsigned i5 = ax[5].closestValidBin(x5);
        const unsigned i6 = ax[6].closestValidBin(x6);
        const unsigned i7 = ax[7].closestValidBin(x7);
        return data_(i0, i1, i2, i3, i4, i5, i6, i7);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const double x3,
                                     const double x4, const double x5,
                                     const double x6, const double x7,
                                     const double x8, const Num2& w)
    {
        if (dim_ != 9U) Private::h_badargs("fill");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0, i8=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        const unsigned o8 = ax[8].overflowIndex(x8, &i8);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U)
            data_(i0, i1, i2, i3, i4, i5, i6, i7, i8) += w;
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, const double x3,
                                         const double x4, const double x5,
                                         const double x6, const double x7,
                                         const double x8, Num2& w, Functor& f)
    {
        if (dim_ != 9U) Private::h_badargs("dispatch");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0, i8=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        const unsigned o8 = ax[8].overflowIndex(x8, &i8);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U)
            f(data_(i0, i1, i2, i3, i4, i5, i6, i7, i8), w);
        else
            f(overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const double x2, const double x3,
                                      const double x4, const double x5,
                                      const double x6, const double x7,
                                      const double x8, const Num2& w)
    {
        if (dim_ != 9U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        const unsigned o3 = ax[3].overflowIndexWeighted(x3, idx+3, wg+3);
        const unsigned o4 = ax[4].overflowIndexWeighted(x4, idx+4, wg+4);
        const unsigned o5 = ax[5].overflowIndexWeighted(x5, idx+5, wg+5);
        const unsigned o6 = ax[6].overflowIndexWeighted(x6, idx+6, wg+6);
        const unsigned o7 = ax[7].overflowIndexWeighted(x7, idx+7, wg+7);
        const unsigned o8 = ax[8].overflowIndexWeighted(x8, idx+8, wg+8);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(
        const double x0, const double x1,
        const double x2, const double x3,
        const double x4, const double x5,
        const double x6, const double x7,
        const double x8) const
    {
        if (dim_ != 9U) Private::h_badargs("examine");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0, i8=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        const unsigned o8 = ax[8].overflowIndex(x8, &i8);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U)
            return data_(i0, i1, i2, i3, i4, i5, i6, i7, i8);
        else
            return overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2,
                                                     const double x3,
                                                     const double x4,
                                                     const double x5,
                                                     const double x6,
                                                     const double x7,
                                                     const double x8) const
    {
        if (dim_ != 9U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        const unsigned i3 = ax[3].closestValidBin(x3);
        const unsigned i4 = ax[4].closestValidBin(x4);
        const unsigned i5 = ax[5].closestValidBin(x5);
        const unsigned i6 = ax[6].closestValidBin(x6);
        const unsigned i7 = ax[7].closestValidBin(x7);
        const unsigned i8 = ax[8].closestValidBin(x8);
        return data_(i0, i1, i2, i3, i4, i5, i6, i7, i8);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fill(const double x0, const double x1,
                                     const double x2, const double x3,
                                     const double x4, const double x5,
                                     const double x6, const double x7,
                                     const double x8, const double x9,
                                     const Num2& w)
    {
        if (dim_ != 10U) Private::h_badargs("fill");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0, i8=0, i9=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        const unsigned o8 = ax[8].overflowIndex(x8, &i8);
        const unsigned o9 = ax[9].overflowIndex(x9, &i9);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U && o9 == 1U)
            data_(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9) += w;
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, class Functor>
    void HistoND<Numeric,Axis>::dispatch(const double x0, const double x1,
                                         const double x2, const double x3,
                                         const double x4, const double x5,
                                         const double x6, const double x7,
                                         const double x8, const double x9,
                                         Num2& w, Functor& f)
    {
        if (dim_ != 10U) Private::h_badargs("dispatch");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0, i8=0, i9=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        const unsigned o8 = ax[8].overflowIndex(x8, &i8);
        const unsigned o9 = ax[9].overflowIndex(x9, &i9);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U && o9 == 1U)
            f(data_(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9), w);
        else
            f(overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9), w);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::fillC(const double x0, const double x1,
                                      const double x2, const double x3,
                                      const double x4, const double x5,
                                      const double x6, const double x7,
                                      const double x8, const double x9,
                                      const Num2& w)
    {
        if (dim_ != 10U) Private::h_badargs("fillC");
        double* wg = &weightBuf_[0];
        unsigned* idx = &indexBuf_[0];
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndexWeighted(x0, idx+0, wg+0);
        const unsigned o1 = ax[1].overflowIndexWeighted(x1, idx+1, wg+1);
        const unsigned o2 = ax[2].overflowIndexWeighted(x2, idx+2, wg+2);
        const unsigned o3 = ax[3].overflowIndexWeighted(x3, idx+3, wg+3);
        const unsigned o4 = ax[4].overflowIndexWeighted(x4, idx+4, wg+4);
        const unsigned o5 = ax[5].overflowIndexWeighted(x5, idx+5, wg+5);
        const unsigned o6 = ax[6].overflowIndexWeighted(x6, idx+6, wg+6);
        const unsigned o7 = ax[7].overflowIndexWeighted(x7, idx+7, wg+7);
        const unsigned o8 = ax[8].overflowIndexWeighted(x8, idx+8, wg+8);
        const unsigned o9 = ax[9].overflowIndexWeighted(x9, idx+9, wg+9);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U && o9 == 1U)
            fillPreservingCentroid(w);
        else
        {
            overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9) += w;
            ++overCount_;
        }
        ++fillCount_; ++modCount_;
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::examine(
        const double x0, const double x1,
        const double x2, const double x3,
        const double x4, const double x5,
        const double x6, const double x7,
        const double x8,
        const double x9) const
    {
        if (dim_ != 10U) Private::h_badargs("examine");
        unsigned i0=0, i1=0, i2=0, i3=0, i4=0, i5=0, i6=0, i7=0, i8=0, i9=0;
        const Axis* ax = &axes_[0];
        const unsigned o0 = ax[0].overflowIndex(x0, &i0);
        const unsigned o1 = ax[1].overflowIndex(x1, &i1);
        const unsigned o2 = ax[2].overflowIndex(x2, &i2);
        const unsigned o3 = ax[3].overflowIndex(x3, &i3);
        const unsigned o4 = ax[4].overflowIndex(x4, &i4);
        const unsigned o5 = ax[5].overflowIndex(x5, &i5);
        const unsigned o6 = ax[6].overflowIndex(x6, &i6);
        const unsigned o7 = ax[7].overflowIndex(x7, &i7);
        const unsigned o8 = ax[8].overflowIndex(x8, &i8);
        const unsigned o9 = ax[9].overflowIndex(x9, &i9);
        if (o0 == 1U && o1 == 1U && o2 == 1U && o3 == 1U && o4 == 1U &&
            o5 == 1U && o6 == 1U && o7 == 1U && o8 == 1U && o9 == 1U)
            return data_(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9);
        else
            return overflow_(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9);
    }

    template <typename Numeric, class Axis>
    const Numeric& HistoND<Numeric,Axis>::closestBin(const double x0,
                                                     const double x1,
                                                     const double x2,
                                                     const double x3,
                                                     const double x4,
                                                     const double x5,
                                                     const double x6,
                                                     const double x7,
                                                     const double x8,
                                                     const double x9) const
    {
        if (dim_ != 10U) Private::h_badargs("closestBin");
        const Axis* ax = &axes_[0];
        const unsigned i0 = ax[0].closestValidBin(x0);
        const unsigned i1 = ax[1].closestValidBin(x1);
        const unsigned i2 = ax[2].closestValidBin(x2);
        const unsigned i3 = ax[3].closestValidBin(x3);
        const unsigned i4 = ax[4].closestValidBin(x4);
        const unsigned i5 = ax[5].closestValidBin(x5);
        const unsigned i6 = ax[6].closestValidBin(x6);
        const unsigned i7 = ax[7].closestValidBin(x7);
        const unsigned i8 = ax[8].closestValidBin(x8);
        const unsigned i9 = ax[9].closestValidBin(x9);
        return data_(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned *index,
                                              const unsigned indexLen,
                                              const Num2& v)
    {
        data_.value(index, indexLen) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned *index,
                                                const unsigned indexLen,
                                                const Num2& v)
    {
        data_.valueAt(index, indexLen) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const Num2& v)
    {
        data_() = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const Num2& v)
    {
        data_.at() = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(
        const unsigned i0, const Num2& v)
    {
        data_(i0) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(
        const unsigned i0, const Num2& v)
    {
        data_.at(i0) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const Num2& v)
    {
        data_(i0, i1) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const Num2& v)
    {
        data_.at(i0, i1) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const Num2& v)
    {
        data_(i0, i1, i2) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const unsigned i3,
                                              const Num2& v)
    {
        data_(i0, i1, i2, i3) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const unsigned i3,
                                              const unsigned i4,
                                              const Num2& v)
    {
        data_(i0, i1, i2, i3, i4) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const unsigned i3,
                                              const unsigned i4,
                                              const unsigned i5,
                                              const Num2& v)
    {
        data_(i0, i1, i2, i3, i4, i5) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const unsigned i3,
                                              const unsigned i4,
                                              const unsigned i5,
                                              const unsigned i6,
                                              const Num2& v)
    {
        data_(i0, i1, i2, i3, i4, i5, i6) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const unsigned i3,
                                              const unsigned i4,
                                              const unsigned i5,
                                              const unsigned i6,
                                              const unsigned i7,
                                              const Num2& v)
    {
        data_(i0, i1, i2, i3, i4, i5, i6, i7) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const unsigned i3,
                                              const unsigned i4,
                                              const unsigned i5,
                                              const unsigned i6,
                                              const unsigned i7,
                                              const unsigned i8,
                                              const Num2& v)
    {
        data_(i0, i1, i2, i3, i4, i5, i6, i7, i8) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBin(const unsigned i0,
                                              const unsigned i1,
                                              const unsigned i2,
                                              const unsigned i3,
                                              const unsigned i4,
                                              const unsigned i5,
                                              const unsigned i6,
                                              const unsigned i7,
                                              const unsigned i8,
                                              const unsigned i9,
                                              const Num2& v)
    {
        data_(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const unsigned i3,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2, i3) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const unsigned i3,
                                                const unsigned i4,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2, i3, i4) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const unsigned i3,
                                                const unsigned i4,
                                                const unsigned i5,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2, i3, i4, i5) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const unsigned i3,
                                                const unsigned i4,
                                                const unsigned i5,
                                                const unsigned i6,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2, i3, i4, i5, i6) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const unsigned i3,
                                                const unsigned i4,
                                                const unsigned i5,
                                                const unsigned i6,
                                                const unsigned i7,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2, i3, i4, i5, i6, i7) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const unsigned i3,
                                                const unsigned i4,
                                                const unsigned i5,
                                                const unsigned i6,
                                                const unsigned i7,
                                                const unsigned i8,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2, i3, i4, i5, i6, i7, i8) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinAt(const unsigned i0,
                                                const unsigned i1,
                                                const unsigned i2,
                                                const unsigned i3,
                                                const unsigned i4,
                                                const unsigned i5,
                                                const unsigned i6,
                                                const unsigned i7,
                                                const unsigned i8,
                                                const unsigned i9,
                                                const Num2& v)
    {
        data_.at(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9) = v;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline HistoND<Numeric,Axis>& HistoND<Numeric,Axis>::operator+=(
        const HistoND<Num2,Axis>& r)
    {
        data_ += r.data_;
        overflow_ += r.overflow_;
        fillCount_ += r.fillCount_;
        overCount_ += r.overCount_;
        ++modCount_;
        return *this;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline HistoND<Numeric,Axis>& HistoND<Numeric,Axis>::operator-=(
        const HistoND<Num2,Axis>& r)
    {
        data_ -= r.data_;
        overflow_ -= r.overflow_;

        // Subtraction does not make much sense for fill counts.
        // We will assume that what we want should be equivalent
        // to the in-place multiplication of the other histogram
        // by -1 and then adding.
        //
        fillCount_ += r.fillCount_;
        overCount_ += r.overCount_;

        ++modCount_;
        return *this;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline HistoND<Numeric,Axis>&
    HistoND<Numeric,Axis>::operator*=(const Num2& r)
    {
        data_ *= r;
        overflow_ *= r;
        ++modCount_;
        return *this;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline HistoND<Numeric,Axis>&
    HistoND<Numeric,Axis>::operator/=(const Num2& r)
    {
        data_ /= r;
        overflow_ /= r;
        ++modCount_;
        return *this;
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const HistoND& r,
                                   const unsigned ax1,
                                   const unsigned ax2)
        : title_(r.title_),
          accumulatedDataLabel_(r.accumulatedDataLabel_),
          data_(r.data_.transpose(ax1, ax2)),
          overflow_(r.overflow_.transpose(ax1, ax2)),
          axes_(r.axes_),
          weightBuf_(r.weightBuf_),
          indexBuf_(r.indexBuf_),
          fillCount_(r.fillCount_),
          overCount_(r.overCount_),
          modCount_(0UL),
          dim_(r.dim_)
    {
        std::swap(axes_[ax1], axes_[ax2]);
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>::HistoND(const HistoND& r)
        : title_(r.title_),
          accumulatedDataLabel_(r.accumulatedDataLabel_),
          data_(r.data_),
          overflow_(r.overflow_),
          axes_(r.axes_),
          weightBuf_(r.weightBuf_),
          indexBuf_(r.indexBuf_),
          fillCount_(r.fillCount_),
          overCount_(r.overCount_),
          modCount_(0UL),
          dim_(r.dim_)
    {
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis>& HistoND<Numeric,Axis>::operator=(const HistoND& r)
    {
        if (&r != this)
        {
            title_ = r.title_;
            accumulatedDataLabel_ = r.accumulatedDataLabel_;
            data_.uninitialize();
            data_ = r.data_;
            overflow_.uninitialize();
            overflow_ = r.overflow_;
            axes_ = r.axes_;
            weightBuf_ = r.weightBuf_;
            indexBuf_ = r.indexBuf_;
            fillCount_ = r.fillCount_;
            overCount_ = r.overCount_;
            dim_ = r.dim_;
            ++modCount_;
        }
        return *this;
    }

    template <typename Numeric, class Axis>
    HistoND<Numeric,Axis> HistoND<Numeric,Axis>::transpose(
        const unsigned axisNum1, const unsigned axisNum2) const
    {
        if (axisNum1 >= dim_ || axisNum2 >= dim_)
            throw npstat::NpstatOutOfRange("In npstat::HistoND::transpose: "
                                    "axis number is out of range");
        if (axisNum1 == axisNum2)
            // Just make a copy
            return *this;
        else
            return HistoND(*this, axisNum1, axisNum2);
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::addToBinContents(const Num2& weight)
    {
        const unsigned long nDat = data_.length();
        Numeric* data = const_cast<Numeric*>(data_.data());
        for (unsigned long i=0; i<nDat; ++i)
            data[i] += weight;
        fillCount_ += nDat;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::addToOverflows(const Num2& weight)
    {
        const unsigned long nOver = overflow_.length();
        Numeric* data = const_cast<Numeric*>(overflow_.data());
        for (unsigned long i=0; i<nOver; ++i)
            data[i] += weight;
        overCount_ += nOver;
        fillCount_ += nOver;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::addToBinContents(
        const Num2* data, const unsigned long dataLength)
    {
        if (dataLength != data_.length()) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::addToBinContents: incompatible data length");
        assert(data);
        Numeric* dat = const_cast<Numeric*>(data_.data());
        for (unsigned long i=0; i<dataLength; ++i)
            dat[i] += data[i];
        fillCount_ += dataLength;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::addToOverflows(
        const Num2* data, const unsigned long dataLength)
    {
        if (dataLength != overflow_.length()) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::addToOverflows: incompatible data length");
        assert(data);
        Numeric* dat = const_cast<Numeric*>(overflow_.data());
        for (unsigned long i=0; i<dataLength; ++i)
            dat[i] += data[i];
        overCount_ += dataLength;
        fillCount_ += dataLength;
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::scaleBinContents(
        const Num2* data, const unsigned long dataLength)
    {
        if (dataLength != data_.length()) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::scaleBinContents: incompatible data length");
        assert(data);
        Numeric* dat = const_cast<Numeric*>(data_.data());
        for (unsigned long i=0; i<dataLength; ++i)
            dat[i] *= data[i];
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    void HistoND<Numeric,Axis>::scaleOverflows(
        const Num2* data, const unsigned long dataLength)
    {
        if (dataLength != overflow_.length()) throw npstat::NpstatInvalidArgument(
            "In npstat::HistoND::scaleOverflows: incompatible data length");
        assert(data);
        Numeric* dat = const_cast<Numeric*>(overflow_.data());
        for (unsigned long i=0; i<dataLength; ++i)
            dat[i] *= data[i];
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setBinContents(
        const Num2* data, const unsigned long dataLength,
        const bool clearOverflowsNow)
    {
        data_.setData(data, dataLength);
        if (clearOverflowsNow)
            clearOverflows();
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2>
    inline void HistoND<Numeric,Axis>::setOverflows(
        const Num2* data, const unsigned long dataLength)
    {
        overflow_.setData(data, dataLength);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    inline void HistoND<Numeric,Axis>::recalculateNFillsFromData()
    {
        const long double nOver = overflow_.template sum<long double>();
        const long double nData = data_.template sum<long double>();
        overCount_ = static_cast<unsigned long>(nOver);
        fillCount_ = static_cast<unsigned long>(nData + nOver);
        ++modCount_;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, typename Num3>
    inline void HistoND<Numeric,Axis>::addToProjection(
        HistoND<Num2,Axis>* projection,
        AbsArrayProjector<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        data_.addToProjection(&projection->data_, projector,
                              projectedIndices, nProjectedIndices);
        projection->fillCount_ += projection->nBins();
        projection->modCount_++;
    }

    template <typename Numeric, class Axis>
    template <typename Num2, typename Num3>
    inline void HistoND<Numeric,Axis>::addToProjection(
        HistoND<Num2,Axis>* projection,
        AbsVisitor<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        data_.addToProjection(&projection->data_, projector,
                              projectedIndices, nProjectedIndices);
        projection->fillCount_ += projection->nBins();
        projection->modCount_++;
    }

    template <typename Numeric, class Axis>
    const char* HistoND<Numeric,Axis>::classname()
    {
        static const std::string myClass(gs::template_class_name<Numeric,Axis>(
                                             "npstat::HistoND"));
        return myClass.c_str();
    }

    template<typename Numeric, class Axis>
    bool HistoND<Numeric,Axis>::write(std::ostream& of) const
    {
        gs::write_pod(of, title_);
        gs::write_pod(of, accumulatedDataLabel_);
        gs::write_pod(of, fillCount_);
        gs::write_pod(of, overCount_);

        return !of.fail() &&
            gs::write_obj_vector(of, axes_) && 
            data_.classId().write(of) &&
            data_.write(of) &&
            overflow_.write(of);
    }

    template<typename Numeric, class Axis>
    HistoND<Numeric,Axis>* HistoND<Numeric,Axis>::read(const gs::ClassId& id,
                                                       std::istream& in)
    {
        static const gs::ClassId current(
            gs::ClassId::makeId<HistoND<Numeric,Axis> >());
        current.ensureSameId(id);

        std::string title;
        gs::read_pod(in, &title);

        std::string accumulatedDataLabel;
        gs::read_pod(in, &accumulatedDataLabel);

        unsigned long fillCount = 0, overCount = 0;
        gs::read_pod(in, &fillCount);
        gs::read_pod(in, &overCount);
        if (in.fail()) throw gs::IOReadFailure(
            "In npstat::HistoND::read: input stream failure");

        std::vector<Axis> axes;
        gs::read_heap_obj_vector_as_placed(in, &axes);
        gs::ClassId ida(in, 1);
        ArrayND<Numeric> data, over;
        ArrayND<Numeric>::restore(ida, in, &data);
        ArrayND<Numeric>::restore(ida, in, &over);
        CPP11_auto_ptr<HistoND<Numeric,Axis> > result(
            new HistoND<Numeric,Axis>(axes, title.c_str(),
                                      accumulatedDataLabel.c_str()));
        result->data_ = data;
        result->overflow_ = over;
        result->fillCount_ = fillCount;
        result->overCount_ = overCount;
        return result.release();
    }

    template <typename Histo>
    void convertHistoToDensity(Histo* h, const bool knownNonNegative)
    {
        assert(h);
        if (!knownNonNegative)
            (const_cast<ArrayND<typename 
                Histo::value_type>&>(h->binContents())).makeNonNegative();
        const double integ = h->integral();
        *h /= integ;
    }

    template <typename Histo>
    std::vector<LinearMapper1d> densityScanHistoMap(const Histo& histo)
    {
        std::vector<LinearMapper1d> result;
        const unsigned d = histo.dim();
        result.reserve(d);
        for (unsigned i=0; i<d; ++i)
        {
            const LinearMapper1d& m = histo.axis(i).binNumberMapper(false);
            result.push_back(m.inverse());
        }
        return result;
    }

    template <typename Histo>
    std::vector<CircularMapper1d> convolutionHistoMap(
        const Histo& histo, const bool doubleRange)
    {
        std::vector<CircularMapper1d> result;
        const unsigned d = histo.dim();
        result.reserve(d);
        for (unsigned i=0; i<d; ++i)
            result.push_back(histo.axis(i).kernelScanMapper(doubleRange));
        return result;
    }
}


#endif // NPSTAT_HISTOND_HH_

