#ifndef NPSTAT_ARRAYND_HH_
#define NPSTAT_ARRAYND_HH_

/*!
// \file ArrayND.h
//
// \brief Arbitrary-dimensional array template
//
// Author: I. Volobouev
//
// October 2009
*/

#include <cassert>

#include "Alignment/Geners/interface/ClassId.hh"

#include "JetMETCorrections/InterpolationTables/interface/SimpleFunctors.h"
#include "JetMETCorrections/InterpolationTables/interface/ArrayRange.h"
#include "JetMETCorrections/InterpolationTables/interface/AbsArrayProjector.h"
#include "JetMETCorrections/InterpolationTables/interface/AbsVisitor.h"
#include "JetMETCorrections/InterpolationTables/interface/PreciseType.h"
#include "JetMETCorrections/InterpolationTables/interface/ProperDblFromCmpl.h"

namespace npstat {
    /**
    // A class for multidimensional array manipulation. A number of methods
    // of this class will work only if dimensionality is limited by
    // CHAR_BIT*sizeof(unsigned long)-1 (which is 31 and 63 on 32- and 64-bit
    // architectures, respectively).
    //
    // Depending on how much space is provided with the "StackLen" template
    // parameter, the array data will be placed either on the stack or on the
    // heap. By default, array data leaves on the heap unless the array has
    // rank 0.
    //
    // Depending on how much space is provided with the "StackDim" template
    // parameter, the array strides will be placed either on the stack or
    // on the heap. By default, strides will be placed on the stack in case
    // the array dimensionality is ten or less.
    //
    // The "Numeric" type must have a default constructor (of course,
    // pointers to arbitrary types can be used as well).
    //
    // Both StackLen and StackDim parameters must be positive.
    */
    template <typename Numeric, unsigned StackLen=1U, unsigned StackDim=10U>
    class ArrayND
    {
        template <typename Num2, unsigned Len2, unsigned Dim2>
        friend class ArrayND;

    public:
        typedef Numeric value_type;
        typedef typename ProperDblFromCmpl<Numeric>::type proper_double;

        /**
        // Default constructor creates an uninitialized array.
        // Only three things can be done safely with such an array:
        //
        // 1) Assigning it from another array (initialized or not).
        //
        // 2) Passing it as an argument to the class static method "restore".
        //
        // 3) Calling the "uninitialize" method.
        //
        // Any other operation results in an undefined behavior (often,
        // an exception is thrown). Note that initialized array can not
        // be assigned from uninitialized one.
        */
        ArrayND();

        //@{
        /**
        // Constructor which creates arrays with the given shape.
        // The array data remains undefined. Simple inilitalization
        // of the data can be performed using methods clear() or
        // constFill(SomeValue). More complicated initialization
        // can be done by "linearFill", "functorFill", or by setting
        // every array element to a desired value.
        */
        explicit ArrayND(const ArrayShape& shape);
        ArrayND(const unsigned* shape, unsigned dim);
        //@}

        /** The copy constructor */
        ArrayND(const ArrayND&);

        /**
        // Converting constructor. It looks more general than the copy
        // constructor, but the actual copy constructor has to be created
        // anyway -- otherwise the compiler will generate an incorrect
        // default copy constructor. Note that existence of this
        // constructor essentially disables data type safety for copying
        // arrays -- but the code significantly gains in convenience.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND(const ArrayND<Num2, Len2, Dim2>&);

        /**
        // Converting constructor where the array values are filled
        // by a functor using values of another array as arguments
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        ArrayND(const ArrayND<Num2, Len2, Dim2>&, Functor f);

        /** Constructor from a subrange of another array */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND(const ArrayND<Num2, Len2, Dim2>& from,
                const ArrayRange& fromRange);

        /** Similar constructor with a transforming functor */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        ArrayND(const ArrayND<Num2, Len2, Dim2>& from,
                const ArrayRange& fromRange, Functor f);

        /**
        // Constructor from a slice of another array. The data of the
        // constructed array remains undefined. The argument "indices"
        // lists either the array indices whose numbers will be fixed
        // when slicing is performed or the indices which will be iterated
        // over during projections (for example, array values may be
        // summed over these indices). These indices will be excluded
        // from the constructed array. The created array can be subsequently
        // used with methods "exportSlice", "importSlice", "project", etc.
        // of the parent array "slicedArray".
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND(const ArrayND<Num2, Len2, Dim2>& slicedArray,
                const unsigned *indices, unsigned nIndices);

        /** Outer product constructor */
        template <typename Num1, unsigned Len1, unsigned Dim1,
                  typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND(const ArrayND<Num1, Len1, Dim1>& a1,
                const ArrayND<Num2, Len2, Dim2>& a2);

        //@{
        /** 
        // Constructor in which the spans are explicitly provided
        // for each dimension. The array data remains undefined.
        */
        explicit ArrayND(unsigned n0);
        ArrayND(unsigned n0, unsigned n1);
        ArrayND(unsigned n0, unsigned n1, unsigned n2);
        ArrayND(unsigned n0, unsigned n1, unsigned n2, unsigned n3);
        ArrayND(unsigned n0, unsigned n1, unsigned n2, unsigned n3,
                unsigned n4);
        ArrayND(unsigned n0, unsigned n1, unsigned n2, unsigned n3,
                unsigned n4, unsigned n5);
        ArrayND(unsigned n0, unsigned n1, unsigned n2, unsigned n3,
                unsigned n4, unsigned n5, unsigned n6);
        ArrayND(unsigned n0, unsigned n1, unsigned n2, unsigned n3,
                unsigned n4, unsigned n5, unsigned n6, unsigned n7);
        ArrayND(unsigned n0, unsigned n1, unsigned n2, unsigned n3,
                unsigned n4, unsigned n5, unsigned n6, unsigned n7,
                unsigned n8);
        ArrayND(unsigned n0, unsigned n1, unsigned n2, unsigned n3,
                unsigned n4, unsigned n5, unsigned n6, unsigned n7,
                unsigned n8, unsigned n9);
        //@}

        /** Destructor */
        ~ArrayND();

        /**
        // Assignment operator. The shape of the array on the right
        // must be compatible with the shape of the array on the left.
        // The only exception is when the array on the left has no shape
        // at all (i.e., it was created by the default constructor or
        // its "uninitialize" method was called). In this case the array
        // on the left will assume the shape of the array on the right.
        */
        ArrayND& operator=(const ArrayND&);

        /** Converting assignment operator */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND& operator=(const ArrayND<Num2,Len2,Dim2>&);

        /** Converting assignment method with a transforming functor */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        ArrayND& assign(const ArrayND<Num2, Len2, Dim2>&, Functor f);

        /**
        // The function which can "uninitialize" the array to the same
        // state as produced by the default constructor. Can be applied
        // in order to force the assignment operators to work.
        */
        ArrayND& uninitialize();

        //@{
        /**
        // Element access using multidimensional array index
        // (no bounds checking). The length of the index array
        // must be equal to the rank of this object.
        */
        Numeric& value(const unsigned *index, unsigned indexLen);
        const Numeric& value(const unsigned *index, unsigned indexLen) const;
        //@}

        //@{
        /**
        // Element access using multidimensional array index
        // (with bounds checking)
        */
        Numeric& valueAt(const unsigned *index, unsigned indexLen);
        const Numeric& valueAt(const unsigned *index, unsigned indexLen) const;
        //@}

        //@{
        /** Element access using linear index (no bounds checking) */
        Numeric& linearValue(unsigned long index);
        const Numeric& linearValue(unsigned long index) const;
        //@}

        //@{
        /** Element access using linear index (with bounds checking) */
        Numeric& linearValueAt(unsigned long index);
        const Numeric& linearValueAt(unsigned long index) const;
        //@}

        /** Convert linear index into multidimensional index */
        void convertLinearIndex(unsigned long l, unsigned* index,
                                unsigned indexLen) const;

        /** Convert multidimensional index into linear index */
        unsigned long linearIndex(const unsigned* idx, unsigned idxLen) const;

        // Some inspectors
        /** Total number of data array elements */
        inline unsigned long length() const {return len_;}

        /** Linearized data */
        inline const Numeric* data() const {return data_;}

        /** Check whether the array has been initialized */
        inline bool isShapeKnown() const {return shapeIsKnown_;}

        /** The number of array dimensions */
        inline unsigned rank() const {return dim_;}

        /** Get the complete shape */
        ArrayShape shape() const;

        /** Shape data as a C-style array */
        inline const unsigned *shapeData() const {return shape_;}

        /** Get the complete range */
        ArrayRange fullRange() const;

        /** Get the number of elements in some particular dimension */
        unsigned span(unsigned dim) const;

        /** Maximum span among all dimensions */
        unsigned maximumSpan() const;

        /** Minimum span among all dimensions */
        unsigned minimumSpan() const;

        /** Get the strides */
        inline const unsigned long* strides() const {return strides_;}

        /** Check if all array elements are zero */
        bool isZero() const;

        /**
        // This method checks whether all array elements are
        // non-negative and, in addition, there is at least
        // one positive element
        */
        bool isDensity() const;

        /** This method modifies all the data in one statement */
        template <typename Num2>
        ArrayND& setData(const Num2* data, unsigned long dataLength);

        /** Compare two arrays for equality */
        template <unsigned Len2, unsigned Dim2>
        bool operator==(const ArrayND<Numeric,Len2,Dim2>&) const;

        /** Logical negation of operator== */
        template <unsigned Len2, unsigned Dim2>
        bool operator!=(const ArrayND<Numeric,Len2,Dim2>&) const;

        /** Largest absolute difference with another bin-compatible array */
        template <unsigned Len2, unsigned Dim2>
        double maxAbsDifference(const ArrayND<Numeric,Len2,Dim2>&) const;

        /** operator+ returns a copy of this array */
        ArrayND operator+() const;

        /** operator- applies the unary minus operator to every element */
        ArrayND operator-() const;

        /** addition of two arrays */
        template <unsigned Len2, unsigned Dim2>
        ArrayND operator+(const ArrayND<Numeric,Len2,Dim2>& r) const;

        /** subtraction of two arrays */
        template <unsigned Len2, unsigned Dim2>
        ArrayND operator-(const ArrayND<Numeric,Len2,Dim2>& r) const;

        /** multiplication by a scalar */
        template <typename Num2>
        ArrayND operator*(const Num2& r) const;

        /** division by a scalar */
        template <typename Num2>
        ArrayND operator/(const Num2& r) const;

        //@{
        /**
        // In-place operators. Note that these work faster than the binary
        // versions, i.e., A += B is much faster than A = A + B.
        */
        template <typename Num2>
        ArrayND& operator*=(const Num2& r);

        template <typename Num2>
        ArrayND& operator/=(const Num2& r);

        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND& operator+=(const ArrayND<Num2,Len2,Dim2>& r);

        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND& operator-=(const ArrayND<Num2,Len2,Dim2>& r);
        //@}

        /** This method is equivalent to (but faster than) += r*c */
        template <typename Num3, typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND& addmul(const ArrayND<Num2,Len2,Dim2>& r, const Num3& c);

        /** Outer product as a method (see also the outer product constructor) */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND outer(const ArrayND<Num2,Len2,Dim2>& r) const;

        /**
        // Contraction of a pair of indices. Note that the array length
        // must be the same in both dimensions.
        */
        ArrayND contract(unsigned pos1, unsigned pos2) const;

        /**
        // Here, dot product corresponds to outer product followed
        // by the contraction over two indices -- the last index
        // of this object and the first index of the argument.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND dot(const ArrayND<Num2,Len2,Dim2>& r) const;

        /**
        // The intent of this method is to marginalize
        // over a set of indices with a prior. Essentially, we are
        // calculating integrals akin to p(y) = Integral f(y|x) g(x) dx
        // in which all functions are represented on an equidistant grid.
        // If needed, multiplication of the result by the grid cell size
        // should be performed after this function. "indexMap" specifies
        // how the indices of the prior array (which is like g(x)) are
        // mapped into the indices of this array (which is like f(y|x)).
        // The number of elements in the map, "mapLen", must be equal to
        // the rank of the prior. Dimension 0 of the prior corresponds
        // to the dimension indexMap[0] of this array, dimension 1
        // corresponds to indexMap[1], etc.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        ArrayND marginalize(const ArrayND<Num2,Len2,Dim2>& prior,
                            const unsigned* indexMap, unsigned mapLen) const;

        /** Transposed array */
        ArrayND transpose(unsigned pos1, unsigned pos2) const;

        /** Transpose without arguments can be invoked for 2-d arrays only */
        ArrayND transpose() const;

        /**
        // Sum of all array elements which uses Num2 type as accumulator.
        // Typically, the precision and dynamic range of Num2 should be
        // suitably larger than the precision and dynamic range of Numeric.
        // For example, if Numeric is float then Num2 should be double, etc.
        */
        template <typename Num2>
        Num2 sum() const;

        /**
        // Sum of absolute values squared which uses Num2 as accumulator.
        // Function std::abs(Numeric) must exist.
        */
        template <typename Num2>
        Num2 sumsq() const;

        /**
        // Mixed derivative over all directions. Useful for generating
        // densities from distribution functions. The resulting array
        // will have one less point in each dimension. Class Num2 is
        // used as accumulator for calculations. static_cast from
        // Num2 to Numeric must exist. The result is multiplied by the
        // scale factor provided.
        */
        template <typename Num2>
        ArrayND derivative(double scale=1.0) const;

        /**
        // The operation inverse to "derivative". Constructs multivariate
        // cumulative density function.
        */
        template <typename Num2>
        ArrayND cdfArray(double scale=1.0) const;

        /**
        // Calculate just one multivariate cumulative density function
        // value. Point with given index will be included in the sum.
        */
        template <typename Num2>
        Num2 cdfValue(const unsigned *index, unsigned indexLen) const;

        /**
        // The next function turns the array data into the conditional
        // cumulative density function for the last dimension. "Num2"
        // is the type of accumulator class used. The cdf is stored
        // in such a way that the cdf value of 0 is skipped (the first
        // stored value is the sum which includes the 0th bin). The slice
        // is filled with the sum of values. The "useTrapezoids" parameter
        // specifies whether trapezoidal integration formula should be
        // utilized (rectangular integration is used in case
        // "useTrapezoids" value is "false").
        */
        template <typename Num2>
        void convertToLastDimCdf(ArrayND* sumSlice, bool useTrapezoids);

        /** Minimum array element */
        Numeric min() const;

        /** Minimum array element and its index */
        Numeric min(unsigned *index, unsigned indexLen) const;

        /** Maximum array element */
        Numeric max() const;

        /** Maximum array element and its index */
        Numeric max(unsigned *index, unsigned indexLen) const;

        //@{
        /**
        // Closest value accessor (works as if the array allows access
        // with non-integer indices). For example, the second point
        // in some dimension will be accessed in case the coordinate
        // in that dimension is between 0.5 and 1.5. This function can be
        // used, for example, for implementing simple N-D histogramming
        // or for closest value interpolation and extrapolation.
        */
        Numeric& closest(const double *x, unsigned xDim);
        const Numeric& closest(const double *x, unsigned xDim) const;
        //@}

        /**
        // Multilinear interpolation. Closest value extrapolation is used
        // in case some index is outside of the array bounds. Note that
        // this function works only if the array dimensionality is less
        // than CHAR_BIT*sizeof(unsigned long). x is the "coordinate"
        // which coincides with array index for x equal to unsigned
        // integers.
        */
        Numeric interpolate1(const double *x, unsigned xDim) const;

        /**
        // Multicubic interpolation. Closest value extrapolation is used
        // in case some index is outside of the array bounds. This
        // function is much slower than "interpolate1" (in the current
        // implementation, a recursive algorithm is used).
        */
        Numeric interpolate3(const double *x, unsigned xDim) const;

        /**
        // This method applies a single-argument functor to each
        // element of the array (in-place). The result returned
        // by the functor becomes the new value of the element. There
        // must be a conversion (static cast) from the functor result to
        // the "Numeric" type. The method returns *this which allows
        // for chaining of such methods. Use the transforming constructor
        // if you want a new array instead.
        */
        template <class Functor>
        ArrayND& apply(Functor f);

        /**
        // This method applies a single-argument functor to each
        // element of the array. The result returned by the functor
        // is ignored inside the scan. Depending on what the functor does,
        // the array values may or may not be modified (they can be modified
        // if the functor takes its argument via a non-const reference).
        */
        template <class Functor>
        ArrayND& scanInPlace(Functor f);

        /** This method fills the array data with a constant value */
        ArrayND& constFill(Numeric c);

        /** Zero the array out (every datum becomes Numeric()) */
        ArrayND& clear();

        /**
        // This method fills the array with a linear combination
        // of the index values. For example, a 2-d array element with indices
        // i, k will be set to (coeff[0]*i + coeff[1]*k + c). There must be
        // a conversion (static cast) from double into "Numeric".
        */
        ArrayND& linearFill(const double* coeff, unsigned coeffLen, double c);

        /**
        // This method fills the array from a functor
        // which takes (const unsigned* index, unsigned indexLen)
        // arguments. There must be a conversion (static cast) from
        // the functor result to the "Numeric" type.
        */
        template <class Functor>
        ArrayND& functorFill(Functor f);

        /**
        // This method can be used for arrays with rank
        // of at least 2 whose length is the same in all dimensions.
        // It puts static_cast<Numeric>(1) on the main diagonal and
        // Numeric() everywhere else.
        */
        ArrayND& makeUnit();

        /** This method turns all negative elements into zeros */
        ArrayND& makeNonNegative();

        /**
        // This method accumulates marginals and divides
        // the array (treated as a distribution) by the product of the
        // marginals. Several iterations like this turn the distribution
        // into a copula. If the array contains negative elements, they
        // are turned into zeros before the iterations are performed.
        // The function returns the actual number of iteration performed
        // when the given tolerance was reached for all marginals.
        */
        unsigned makeCopulaSteps(double tolerance, unsigned maxIterations);

        /**
        // Loop over all elements of two compatible arrays and apply
        // a binary functor
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void jointScan(ArrayND<Num2, Len2, Dim2>& other, Functor binaryFunct);

        /** Convenience method for element-by-element in-place multiplication
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        inline ArrayND& inPlaceMul(const ArrayND<Num2,Len2,Dim2>& r)
        {
            jointScan(const_cast<ArrayND<Num2,Len2,Dim2>&>(r),
                      multeq_left<Numeric,Num2>());
            return *this;
        }

        /**
        // Loop over subranges in two arrays in such a way that the functor
        // is called only if the indices on both sides are valid. The topology
        // of both arrays is assumed to be box-like (flat). The starting
        // corner in this object (where cycling begins) is provided by the
        // argument "thisCorner". The "range" argument specifies the width
        // of the processed patch in each dimension. The corner of the "other"
        // array where cycling begins is provided by the "otherCorner"
        // argument. The "arrLen" argument specifies the number of elements
        // in "thisCorner", "range", and "otherCorner" arrays. It should be
        // equal to the rank of either of the two ArrayND arrays.
        //
        // Note that there is no good way for this method to assume constness
        // of this or "other" array: this becomes apparent only after the 
        // functor has been specified. Apply const_cast judiciously as needed,
        // other solutions of this problem are not any better.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void jointSubrangeScan(ArrayND<Num2, Len2, Dim2>& other,
                               const unsigned* thisCorner,
                               const unsigned* range,
                               const unsigned* otherCorner,
                               unsigned arrLen,
                               Functor binaryFunct);

        /**
        // Method similar to "jointSubrangeScan" in which the topology of
        // both arrays is assumed to be hypertoroidal (circular buffer in
        // every dimension)
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void dualCircularScan(ArrayND<Num2, Len2, Dim2>& other,
                              const unsigned* thisCorner,
                              const unsigned* range,
                              const unsigned* otherCorner,
                              unsigned arrLen,
                              Functor binaryFunct);

        /**
        // Method similar to "jointSubrangeScan" in which the topology of
        // this array is assumed to be flat and the other array hypertoroidal
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void flatCircularScan(ArrayND<Num2, Len2, Dim2>& other,
                              const unsigned* thisCorner,
                              const unsigned* range,
                              const unsigned* otherCorner,
                              unsigned arrLen,
                              Functor binaryFunct);

        /**
        // Method similar to "jointSubrangeScan" in which the topology of
        // this array is assumed to be hypertoroidal and the other array flat
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void circularFlatScan(ArrayND<Num2, Len2, Dim2>& other,
                              const unsigned* thisCorner,
                              const unsigned* range,
                              const unsigned* otherCorner,
                              unsigned arrLen,
                              Functor binaryFunct);

        /**
        // This method runs over a subrange of the array
        // and calls the argument functor on every point. This
        // method will not call "clear" or "result" functions of
        // the argument functor.
        */
        template <typename Num2, typename Integer>
        void processSubrange(AbsArrayProjector<Numeric,Num2>& f,
                             const BoxND<Integer>& subrange) const;

        /**
        // Copy a hyperrectangular subrange of this array potentially
        // completely overwriting the destination array. The starting
        // corner in this object where copying begins is provided by
        // the first two arguments. The subrange size is defined by
        // the shape of the destination array.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        void exportSubrange(const unsigned* fromCorner, unsigned lenCorner,
                            ArrayND<Num2, Len2, Dim2>* dest) const;

        /** The inverse operation to "exportSubrange" */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        void importSubrange(const unsigned* fromCorner, unsigned lenCorner,
                            const ArrayND<Num2, Len2, Dim2>& from);

        /**
        // Check that all elements of this array differ from the
        // corresponding elements of another array by at most "eps".
        // Equivalent to maxAbsDifference(r) <= eps (but usually faster).
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        bool isClose(const ArrayND<Num2,Len2,Dim2>& r, double eps) const;

        /** Check compatibility with another shape */
        bool isCompatible(const ArrayShape& shape) const;

        /**
        // Check shape compatibility with another array. Equivalent to
        // but faster than isCompatible(r.shape()).
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        bool isShapeCompatible(const ArrayND<Num2,Len2,Dim2>& r) const;

        /**
        // Joint cycle over the data of this array and the slice.
        // The array to which the "slice" argument refers should normally
        // be created by the slicing constructor using this array as
        // the argument. The "fixedIndices" argument should be the same
        // as the "indices" argument in that constructor. This method
        // is to be used for import/export of slice data and in-place
        // operations (addition, multiplication, etc).
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void jointSliceScan(ArrayND<Num2,Len2,Dim2>& slice,
                            const unsigned *fixedIndices,
                            const unsigned *fixedIndexValues,
                            unsigned nFixedIndices,
                            Functor binaryFunct);

        /** Convenience method for exporting a slice of this array */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        inline void exportSlice(ArrayND<Num2,Len2,Dim2>* slice,
                                const unsigned *fixedIndices,
                                const unsigned *fixedIndexValues,
                                unsigned nFixedIndices) const
        {
            assert(slice);
            (const_cast<ArrayND*>(this))->jointSliceScan(
                *slice, fixedIndices, fixedIndexValues, nFixedIndices,
                scast_assign_right<Numeric,Num2>());
        }

        /** Convenience method for importing a slice into this array */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        inline void importSlice(const ArrayND<Num2,Len2,Dim2>& slice,
                                const unsigned *fixedIndices,
                                const unsigned *fixedIndexValues,
                                unsigned nFixedIndices)
        {
            jointSliceScan(const_cast<ArrayND<Num2,Len2,Dim2>&>(slice),
                           fixedIndices, fixedIndexValues, nFixedIndices,
                           scast_assign_left<Numeric,Num2>());
        }

        /**
        // This method applies the values in the slice
        // to all other coresponding values in the array. This can
        // be used, for example, to multiply/divide by some factor which
        // varies across the slice. The slice values will be used as
        // the right functor argument.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void applySlice(ArrayND<Num2,Len2,Dim2>& slice,
                        const unsigned *fixedIndices, unsigned nFixedIndices,
                        Functor binaryFunct);

        /**
        // Convenience method which multiplies the array by a scale factor
        // which varies across the slice
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        inline ArrayND& multiplyBySlice(const ArrayND<Num2,Len2,Dim2>& slice,
                                        const unsigned *fixedIndices,
                                        unsigned nFixedIndices)
        {
            applySlice(const_cast<ArrayND<Num2,Len2,Dim2>&>(slice),
                       fixedIndices, nFixedIndices,
                       multeq_left<Numeric,Num2>());
            return *this;
        }

        //@{
        /**
        // This method fills a projection. The array to which
        // "projection" argument points should normally be created by
        // the slicing constructor using this array as an argument.
        // "projectedIndices" should be the same as "indices" specified
        // during the slice creation.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
        void project(ArrayND<Num2,Len2,Dim2>* projection,
                     AbsArrayProjector<Numeric,Num3>& projector,
                     const unsigned *projectedIndices,
                     unsigned nProjectedIndices) const;

        template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
        void project(ArrayND<Num2,Len2,Dim2>* projection,
                     AbsVisitor<Numeric,Num3>& projector,
                     const unsigned *projectedIndices,
                     unsigned nProjectedIndices) const;
        //@}

        //@{
        /**
        // Similar method to "project", but projections are added to
        // (or subtracted from) the existing projection data instead of
        // replacing them
        */
        template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
        void addToProjection(ArrayND<Num2,Len2,Dim2>* projection,
                             AbsArrayProjector<Numeric,Num3>& projector,
                             const unsigned *projectedIndices,
                             unsigned nProjectedIndices) const;

        template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
        void subtractFromProjection(ArrayND<Num2,Len2,Dim2>* projection,
                                    AbsArrayProjector<Numeric,Num3>& projector,
                                    const unsigned *projectedIndices,
                                    unsigned nProjectedIndices) const;

        template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
        void addToProjection(ArrayND<Num2,Len2,Dim2>* projection,
                             AbsVisitor<Numeric,Num3>& projector,
                             const unsigned *projectedIndices,
                             unsigned nProjectedIndices) const;

        template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
        void subtractFromProjection(ArrayND<Num2,Len2,Dim2>* projection,
                                    AbsVisitor<Numeric,Num3>& projector,
                                    const unsigned *projectedIndices,
                                    unsigned nProjectedIndices) const;
        //@}

        /**
        // Rotation. Place the result into another array. The elements
        // with indices 0 in the current array will become elements with
        // indices "shifts" in the rotated array.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        void rotate(const unsigned* shifts, unsigned lenShifts,
                    ArrayND<Num2, Len2, Dim2>* rotated) const;

        /**
        // Fill another array with all possible mirror images
        // of this one. This other array must have twice the span
        // in each dimension.
        */
        template <typename Num2, unsigned Len2, unsigned Dim2>
        void multiMirror(ArrayND<Num2, Len2, Dim2>* out) const;

        //@{
        /**
        // Fortran-style subscripting without bounds checking (of course,
        // with indices starting at 0).
        */
        Numeric& operator()();
        const Numeric& operator()() const;

        Numeric& operator()(unsigned i0);
        const Numeric& operator()(unsigned i0) const;

        Numeric& operator()(unsigned i0, unsigned i1);
        const Numeric& operator()(unsigned i0, unsigned i1) const;

        Numeric& operator()(unsigned i0, unsigned i1, unsigned i2);
        const Numeric& operator()(unsigned i0, unsigned i1, unsigned i2) const;

        Numeric& operator()(unsigned i0, unsigned i1,
                            unsigned i2, unsigned i3);
        const Numeric& operator()(unsigned i0, unsigned i1,
                                  unsigned i2, unsigned i3) const;

        Numeric& operator()(unsigned i0, unsigned i1,
                            unsigned i2, unsigned i3, unsigned i4);
        const Numeric& operator()(unsigned i0, unsigned i1,
                                  unsigned i2, unsigned i3, unsigned i4) const;

        Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                            unsigned i3, unsigned i4, unsigned i5);
        const Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                                  unsigned i3, unsigned i4, unsigned i5) const;

        Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                            unsigned i3, unsigned i4, unsigned i5,
                            unsigned i6);
        const Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                                  unsigned i3, unsigned i4, unsigned i5,
                                  unsigned i6) const;

        Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                            unsigned i3, unsigned i4, unsigned i5,
                            unsigned i6, unsigned i7);
        const Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                                  unsigned i3, unsigned i4, unsigned i5,
                                  unsigned i6, unsigned i7) const;

        Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                            unsigned i3, unsigned i4, unsigned i5,
                            unsigned i6, unsigned i7, unsigned i8);
        const Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                                  unsigned i3, unsigned i4, unsigned i5,
                                  unsigned i6, unsigned i7, unsigned i8) const;

        Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                            unsigned i3, unsigned i4, unsigned i5,
                            unsigned i6, unsigned i7, unsigned i8,
                            unsigned i9);
        const Numeric& operator()(unsigned i0, unsigned i1, unsigned i2,
                                  unsigned i3, unsigned i4, unsigned i5,
                                  unsigned i6, unsigned i7, unsigned i8,
                                  unsigned i9) const;
        //@}

        //@{
        /**
        // Fortran-style subscripting with bounds checking (of course,
        // with indices starting at 0).
        */
        Numeric& at();
        const Numeric& at() const;

        Numeric& at(unsigned i0);
        const Numeric& at(unsigned i0) const;

        Numeric& at(unsigned i0, unsigned i1);
        const Numeric& at(unsigned i0, unsigned i1) const;

        Numeric& at(unsigned i0, unsigned i1, unsigned i2);
        const Numeric& at(unsigned i0, unsigned i1, unsigned i2) const;

        Numeric& at(unsigned i0, unsigned i1,
                    unsigned i2, unsigned i3);
        const Numeric& at(unsigned i0, unsigned i1,
                          unsigned i2, unsigned i3) const;

        Numeric& at(unsigned i0, unsigned i1,
                    unsigned i2, unsigned i3, unsigned i4);
        const Numeric& at(unsigned i0, unsigned i1,
                          unsigned i2, unsigned i3, unsigned i4) const;

        Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                    unsigned i3, unsigned i4, unsigned i5);
        const Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                          unsigned i3, unsigned i4, unsigned i5) const;

        Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                    unsigned i3, unsigned i4, unsigned i5,
                    unsigned i6);
        const Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                          unsigned i3, unsigned i4, unsigned i5,
                          unsigned i6) const;

        Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                    unsigned i3, unsigned i4, unsigned i5,
                    unsigned i6, unsigned i7);
        const Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                          unsigned i3, unsigned i4, unsigned i5,
                          unsigned i6, unsigned i7) const;

        Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                    unsigned i3, unsigned i4, unsigned i5,
                    unsigned i6, unsigned i7, unsigned i8);
        const Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                          unsigned i3, unsigned i4, unsigned i5,
                          unsigned i6, unsigned i7, unsigned i8) const;

        Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                    unsigned i3, unsigned i4, unsigned i5,
                    unsigned i6, unsigned i7, unsigned i8,
                    unsigned i9);
        const Numeric& at(unsigned i0, unsigned i1, unsigned i2,
                          unsigned i3, unsigned i4, unsigned i5,
                          unsigned i6, unsigned i7, unsigned i8,
                          unsigned i9) const;
        //@}

        //@{
        /**
        // Subscripting by continuous coordinate.
        // Works similar to the "closest" method.
        */
        Numeric& cl();
        const Numeric& cl() const;

        Numeric& cl(double x0);
        const Numeric& cl(double x0) const;

        Numeric& cl(double x0, double x1);
        const Numeric& cl(double x0, double x1) const;

        Numeric& cl(double x0, double x1, double x2);
        const Numeric& cl(double x0, double x1, double x2) const;

        Numeric& cl(double x0, double x1,
                    double x2, double x3);
        const Numeric& cl(double x0, double x1,
                          double x2, double x3) const;

        Numeric& cl(double x0, double x1,
                    double x2, double x3, double x4);
        const Numeric& cl(double x0, double x1,
                          double x2, double x3, double x4) const;

        Numeric& cl(double x0, double x1, double x2,
                    double x3, double x4, double x5);
        const Numeric& cl(double x0, double x1, double x2,
                          double x3, double x4, double x5) const;

        Numeric& cl(double x0, double x1, double x2,
                    double x3, double x4, double x5,
                    double x6);
        const Numeric& cl(double x0, double x1, double x2,
                          double x3, double x4, double x5,
                          double x6) const;

        Numeric& cl(double x0, double x1, double x2,
                    double x3, double x4, double x5,
                    double x6, double x7);
        const Numeric& cl(double x0, double x1, double x2,
                          double x3, double x4, double x5,
                          double x6, double x7) const;

        Numeric& cl(double x0, double x1, double x2,
                    double x3, double x4, double x5,
                    double x6, double x7, double x8);
        const Numeric& cl(double x0, double x1, double x2,
                          double x3, double x4, double x5,
                          double x6, double x7, double x8) const;

        Numeric& cl(double x0, double x1, double x2,
                    double x3, double x4, double x5,
                    double x6, double x7, double x8,
                    double x9);
        const Numeric& cl(double x0, double x1, double x2,
                          double x3, double x4, double x5,
                          double x6, double x7, double x8,
                          double x9) const;
        //@}

        //@{
        /** Methods related to "geners" I/O */
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static const char* classname();
        static inline unsigned version() {return 1;}
        static void restore(const gs::ClassId& id, std::istream& in,
                            ArrayND* array);
    private:
        Numeric localData_[StackLen];
        Numeric* data_;

        unsigned long localStrides_[StackDim];
        unsigned long *strides_;

        unsigned localShape_[StackDim];
        unsigned *shape_;

        unsigned long len_;
        unsigned dim_;

        bool shapeIsKnown_;

        // Basic initialization from unsigned* shape and dimensionality
        void buildFromShapePtr(const unsigned*, unsigned);

        // Build strides_ array out of the shape_ array
        void buildStrides();

        // Recursive implementation of nested loops for "linearFill"
        void linearFillLoop(unsigned level, double s0,
                            unsigned long idx, double shift,
                            const double* coeffs);

        // Recursive implementation of nested loops for "functorFill"
        template <class Functor>
        void functorFillLoop(unsigned level, unsigned long idx,
                             Functor f, unsigned* farg);

        // Recursive implementation of nested loops for "interpolate3"
        Numeric interpolateLoop(unsigned level, const double *x,
                                const Numeric* base) const;

        // Recursive implementation of nested loops for the outer product
        template <typename Num1, unsigned Len1, unsigned Dim1,
                  typename Num2, unsigned Len2, unsigned Dim2>
        void outerProductLoop(unsigned level, unsigned long idx0,
                              unsigned long idx1, unsigned long idx2,
                              const ArrayND<Num1, Len1, Dim1>& a1,
                              const ArrayND<Num2, Len2, Dim2>& a2);

        // Recursive implementation of nested loops for contraction
        void contractLoop(unsigned thisLevel, unsigned resLevel,
                          unsigned pos1, unsigned pos2,
                          unsigned long idxThis, unsigned long idxRes,
                          ArrayND& result) const;

        // Recursive implementation of nested loops for transposition
        void transposeLoop(unsigned level, unsigned pos1, unsigned pos2,
                           unsigned long idxThis, unsigned long idxRes,
                           ArrayND& result) const;

        // Recursive implementation of nested loops for the dot product
        template <typename Num2, unsigned Len2, unsigned Dim2>
        void dotProductLoop(unsigned level, unsigned long idx0,
                            unsigned long idx1, unsigned long idx2,
                            const ArrayND<Num2, Len2, Dim2>& r,
                            ArrayND& result) const;

        // Recursive implementation of nested loops for marginalization
        template <typename Num2, unsigned Len2, unsigned Dim2>
        Numeric marginalizeInnerLoop(unsigned long idx,
                                     unsigned levelPr, unsigned long idxPr,
                                     const ArrayND<Num2,Len2,Dim2>& prior,
                                     const unsigned* indexMap) const;
        template <typename Num2, unsigned Len2, unsigned Dim2>
        void marginalizeLoop(unsigned level, unsigned long idx,
                             unsigned levelRes, unsigned long idxRes,
                             const ArrayND<Num2,Len2,Dim2>& prior,
                             const unsigned* indexMap, ArrayND& res) const;

        // Recursive implementation of nested loops for range copy
        // with functor modification of elements
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void copyRangeLoopFunct(unsigned level, unsigned long idx0,
                                unsigned long idx1,
                                const ArrayND<Num2, Len2, Dim2>& r,
                                const ArrayRange& range, Functor f);

        // Loop over subrange in such a way that the functor is called
        // only if indices on both sides are valid. The topology of both
        // arrays is that of the hyperplane (flat).
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void commonSubrangeLoop(unsigned level, unsigned long idx0,
                                unsigned long idx1,
                                const unsigned* thisCorner,
                                const unsigned* range,
                                const unsigned* otherCorner,
                                ArrayND<Num2, Len2, Dim2>& other,
                                Functor binaryFunct);

        // Similar loop with the topology of the hypertorus for both
        // arrays (all indices of both arrays are wrapped around)
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void dualCircularLoop(unsigned level, unsigned long idx0,
                              unsigned long idx1,
                              const unsigned* thisCorner,
                              const unsigned* range,
                              const unsigned* otherCorner,
                              ArrayND<Num2, Len2, Dim2>& other,
                              Functor binaryFunct);

        // Similar loop in which the topology of this array is assumed
        // to be flat and the topology of the destination array is that
        // of the hypertorus. Due to the asymmetry of the topologies,
        // "const" function is not provided (use const_cast as appropriate).
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void flatCircularLoop(unsigned level, unsigned long idx0,
                              unsigned long idx1,
                              const unsigned* thisCorner,
                              const unsigned* range,
                              const unsigned* otherCorner,
                              ArrayND<Num2, Len2, Dim2>& other,
                              Functor binaryFunct);

        // Similar loop in which the topology of this array is assumed
        // to be hypertoroidal and the topology of the destination array
        // is flat.
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void circularFlatLoop(unsigned level, unsigned long idx0,
                              unsigned long idx1,
                              const unsigned* thisCorner,
                              const unsigned* range,
                              const unsigned* otherCorner,
                              ArrayND<Num2, Len2, Dim2>& other,
                              Functor binaryFunct);

        // Slice compatibility verification
        template <typename Num2, unsigned Len2, unsigned Dim2>
        unsigned long verifySliceCompatibility(
            const ArrayND<Num2,Len2,Dim2>& slice,
            const unsigned *fixedIndices,
            const unsigned *fixedIndexValues,
            unsigned nFixedIndices) const;

        // Recursive implementation of nested loops for slice operations
        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void jointSliceLoop(unsigned level, unsigned long idx0,
                            unsigned level1, unsigned long idx1,
                            ArrayND<Num2,Len2,Dim2>& slice,
                            const unsigned *fixedIndices,
                            const unsigned *fixedIndexValues,
                            unsigned nFixedIndices, Functor binaryFunctor);

        // Recursive implementation of nested loops for "applySlice"
        template <typename Num2, class Functor>
        void scaleBySliceInnerLoop(unsigned level, unsigned long idx0,
                                   Num2& scale,
                                   const unsigned* projectedIndices,
                                   unsigned nProjectedIndices,
                                   Functor binaryFunct);

        template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
        void scaleBySliceLoop(unsigned level, unsigned long idx0,
                              unsigned level1, unsigned long idx1,
                              ArrayND<Num2,Len2,Dim2>& slice,
                              const unsigned *fixedIndices,
                              unsigned nFixedIndices,
                              Functor binaryFunct);

        // Recursive implementation of nested loops for projections
        template <typename Num2>
        void projectInnerLoop(unsigned level, unsigned long idx0,
                              unsigned* currentIndex,
                              AbsArrayProjector<Numeric,Num2>& projector,
                              const unsigned* projectedIndices,
                              unsigned nProjectedIndices) const;

        template <typename Num2, unsigned Len2, unsigned Dim2,
                  typename Num3, class Op>
        void projectLoop(unsigned level, unsigned long idx0,
                         unsigned level1, unsigned long idx1,
                         unsigned* currentIndex,
                         ArrayND<Num2,Len2,Dim2>* projection,
                         AbsArrayProjector<Numeric,Num3>& projector,
                         const unsigned* projectedIndices,
                         unsigned nProjectedIndices, Op fcn) const;

        // Note that "projectLoop2" is almost identical to "projectLoop"
        // while "projectInnerLoop2" is almost identical to "projectInnerLoop".
        // It would make a lot of sense to combine these functions into
        // the same code and then partially specialize the little piece
        // where the "AbsVisitor" or "AbsArrayProjector" is actually called.
        // Unfortunately, "AbsVisitor" and "AbsArrayProjector" are
        // templates themselves, and it is not possible in C++ to partially
        // specialize a function template (that is, even if we can specialize
        // on "AbsVisitor" vs. "AbsArrayProjector", there is no way to
        // specialize on their parameter types).
        template <typename Num2, unsigned Len2, unsigned Dim2,
                  typename Num3, class Op>
        void projectLoop2(unsigned level, unsigned long idx0,
                          unsigned level1, unsigned long idx1,
                          ArrayND<Num2,Len2,Dim2>* projection,
                          AbsVisitor<Numeric,Num3>& projector,
                          const unsigned* projectedIndices,
                          unsigned nProjectedIndices, Op fcn) const;

        template <typename Num2>
        void projectInnerLoop2(unsigned level, unsigned long idx0,
                               AbsVisitor<Numeric,Num2>& projector,
                               const unsigned* projectedIndices,
                               unsigned nProjectedIndices) const;

        template <typename Num2, typename Integer>
        void processSubrangeLoop(unsigned level, unsigned long idx0,
                                 unsigned* currentIndex,
                                 AbsArrayProjector<Numeric,Num2>& f,
                                 const BoxND<Integer>& subrange) const;

        // Sum of all points with the given index and below
        template <typename Accumulator>
        Accumulator sumBelowLoop(unsigned level, unsigned long idx0,
                                 const unsigned* limit) const;

        // Loop for "convertToLastDimCdf"
        template <typename Accumulator>
        void convertToLastDimCdfLoop(ArrayND* sumSlice, unsigned level,
                                     unsigned long idx0,
                                     unsigned long idxSlice,
                                     bool useTrapezoids);

        // Convert a coordinate into index.
        // No checking whether "idim" is within limits.
        unsigned coordToIndex(double coord, unsigned idim) const;

        // Verify that projection array is compatible with this one
        template <typename Num2, unsigned Len2, unsigned Dim2>
        void verifyProjectionCompatibility(
            const ArrayND<Num2,Len2,Dim2>& projection,
            const unsigned *projectedIndices,
            unsigned nProjectedIndices) const;

    };
}

#include <cmath>
#include <climits>
#include <algorithm>
#include <sstream>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "Alignment/Geners/interface/GenericIO.hh"
#include "Alignment/Geners/interface/IOIsUnsigned.hh"

#include "JetMETCorrections/InterpolationTables/interface/allocators.h"

#include "JetMETCorrections/InterpolationTables/interface/interpolate.h"
#include "JetMETCorrections/InterpolationTables/interface/absDifference.h"
#include "JetMETCorrections/InterpolationTables/interface/ComplexComparesFalse.h"
#include "JetMETCorrections/InterpolationTables/interface/ComplexComparesAbs.h"

#define me_macro_check_loop_prerequisites(METHOD, INNERLOOP) /**/            \
    template<typename Numeric, unsigned Len, unsigned Dim>                   \
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>    \
    void ArrayND<Numeric,Len,Dim>:: METHOD (                                 \
        ArrayND<Num2, Len2, Dim2>& other,                                    \
        const unsigned* thisCorner,                                          \
        const unsigned* range,                                               \
        const unsigned* otherCorner,                                         \
        const unsigned arrLen,                                               \
        Functor binaryFunct)                                                 \
    {                                                                        \
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(                     \
            "Initialize npstat::ArrayND before calling method \""            \
            #METHOD "\"");                                                   \
        if (!other.shapeIsKnown_) throw npstat::NpstatInvalidArgument(               \
            "In npstat::ArrayND::" #METHOD ": uninitialized argument array");\
        if (dim_ != other.dim_) throw npstat::NpstatInvalidArgument(                 \
            "In npstat::ArrayND::" #METHOD ": incompatible argument array rank");\
        if (arrLen != dim_) throw npstat::NpstatInvalidArgument(                     \
            "In npstat::ArrayND::" #METHOD ": incompatible index length");   \
        if (dim_)                                                            \
        {                                                                    \
            assert(thisCorner);                                              \
            assert(range);                                                   \
            assert(otherCorner);                                             \
            INNERLOOP (0U, 0UL, 0UL, thisCorner, range,                      \
                       otherCorner, other, binaryFunct);                     \
        }                                                                    \
        else                                                                 \
            binaryFunct(localData_[0], other.localData_[0]);                 \
    }

namespace npstat {
    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,Len,Dim>::commonSubrangeLoop(
        unsigned level, unsigned long idx0,
        unsigned long idx1,
        const unsigned* thisCorner,
        const unsigned* range,
        const unsigned* otherCorner,
        ArrayND<Num2, Len2, Dim2>& r,
        Functor binaryFunct)
    {
        const unsigned imax = range[level];

        if (level == dim_ - 1)
        {
            Numeric* left = data_ + (idx0 + thisCorner[level]);
            Numeric* const lMax = data_ + (idx0 + shape_[level]);
            Num2* right = r.data_ + (idx1 + otherCorner[level]);
            Num2* const rMax = r.data_ + (idx1 + r.shape_[level]);

            for (unsigned i=0; i<imax && left<lMax && right<rMax; ++i)
                binaryFunct(*left++, *right++);
        }
        else
        {
            const unsigned long leftStride = strides_[level];
            const unsigned long leftMax = idx0 + shape_[level]*leftStride;
            idx0 += thisCorner[level]*leftStride;
            const unsigned long rightStride = r.strides_[level];
            const unsigned long rightMax = idx1 + r.shape_[level]*rightStride;
            idx1 += otherCorner[level]*rightStride;

            for (unsigned i=0; i<imax && idx0 < leftMax && idx1 < rightMax;
                 ++i, idx0 += leftStride, idx1 += rightStride)
                commonSubrangeLoop(level+1, idx0, idx1, thisCorner, range,
                                   otherCorner, r, binaryFunct);
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,Len,Dim>::dualCircularLoop(
        unsigned level, unsigned long idx0,
        unsigned long idx1,
        const unsigned* thisCorner,
        const unsigned* range,
        const unsigned* otherCorner,
        ArrayND<Num2, Len2, Dim2>& r,
        Functor binaryFunct)
    {
        const unsigned imax = range[level];
        const unsigned leftShift = thisCorner[level];
        const unsigned leftPeriod = shape_[level];
        const unsigned rightShift = otherCorner[level];
        const unsigned rightPeriod = r.shape_[level];

        if (level == dim_ - 1)
        {
            Numeric* left = data_ + idx0;
            Num2* right = r.data_ + idx1;
            for (unsigned i=0; i<imax; ++i)
                binaryFunct(left[(i+leftShift)%leftPeriod],
                            right[(i+rightShift)%rightPeriod]);
        }
        else
        {
            const unsigned long leftStride = strides_[level];
            const unsigned long rightStride = r.strides_[level];
            for (unsigned i=0; i<imax; ++i)
                dualCircularLoop(
                    level+1, idx0+((i+leftShift)%leftPeriod)*leftStride,
                    idx1+((i+rightShift)%rightPeriod)*rightStride,
                    thisCorner, range, otherCorner, r, binaryFunct);
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,Len,Dim>::flatCircularLoop(
        unsigned level, unsigned long idx0,
        unsigned long idx1,
        const unsigned* thisCorner,
        const unsigned* range,
        const unsigned* otherCorner,
        ArrayND<Num2, Len2, Dim2>& r,
        Functor binaryFunct)
    {
        const unsigned imax = range[level];
        const unsigned rightShift = otherCorner[level];
        const unsigned rightPeriod = r.shape_[level];

        if (level == dim_ - 1)
        {
            Numeric* left = data_ + (idx0 + thisCorner[level]);
            Numeric* const lMax = data_ + (idx0 + shape_[level]);
            Num2* right = r.data_ + idx1;

            for (unsigned i=0; i<imax && left<lMax; ++i)
                binaryFunct(*left++, right[(i+rightShift)%rightPeriod]);
        }
        else
        {
            const unsigned long leftStride = strides_[level];
            const unsigned long leftMax = idx0 + shape_[level]*leftStride;
            idx0 += thisCorner[level]*leftStride;
            const unsigned long rightStride = r.strides_[level];

            for (unsigned i=0; i<imax && idx0 < leftMax; ++i, idx0+=leftStride)
                flatCircularLoop(
                    level+1, idx0,
                    idx1+((i+rightShift)%rightPeriod)*rightStride,
                    thisCorner, range, otherCorner, r, binaryFunct);
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,Len,Dim>::circularFlatLoop(
        unsigned level, unsigned long idx0,
        unsigned long idx1,
        const unsigned* thisCorner,
        const unsigned* range,
        const unsigned* otherCorner,
        ArrayND<Num2, Len2, Dim2>& r,
        Functor binaryFunct)
    {
        const unsigned imax = range[level];
        const unsigned leftShift = thisCorner[level];
        const unsigned leftPeriod = shape_[level];

        if (level == dim_ - 1)
        {
            Numeric* left = data_ + idx0;
            Num2* right = r.data_ + (idx1 + otherCorner[level]);
            Num2* const rMax = r.data_ + (idx1 + r.shape_[level]);

            for (unsigned i=0; i<imax && right<rMax; ++i)
                binaryFunct(left[(i+leftShift)%leftPeriod], *right++);
        }
        else
        {
            const unsigned long leftStride = strides_[level];
            const unsigned long rightStride = r.strides_[level];
            const unsigned long rightMax = idx1 + r.shape_[level]*rightStride;
            idx1 += otherCorner[level]*rightStride;

            for (unsigned i=0; i<imax && idx1<rightMax; ++i, idx1+=rightStride)
                circularFlatLoop(
                    level+1, idx0+((i+leftShift)%leftPeriod)*leftStride,
                    idx1, thisCorner, range, otherCorner, r, binaryFunct);
        }
    }

    me_macro_check_loop_prerequisites(jointSubrangeScan, commonSubrangeLoop)
    me_macro_check_loop_prerequisites(dualCircularScan, dualCircularLoop)
    me_macro_check_loop_prerequisites(flatCircularScan, flatCircularLoop)
    me_macro_check_loop_prerequisites(circularFlatScan, circularFlatLoop)

    template <typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    Numeric ArrayND<Numeric,StackLen,StackDim>::marginalizeInnerLoop(
        unsigned long idx, const unsigned levelPr, unsigned long idxPr,
        const ArrayND<Num2,Len2,Dim2>& prior,
        const unsigned* indexMap) const
    {
        Numeric sum = Numeric();
        const unsigned long myStride = strides_[indexMap[levelPr]];
        const unsigned imax = prior.shape_[levelPr];
        assert(imax == shape_[indexMap[levelPr]]);
        if (levelPr == prior.dim_ - 1)
        {
            for (unsigned i=0; i<imax; ++i)
                sum += data_[idx+i*myStride]*prior.data_[idxPr++];
        }
        else
        {
            const unsigned long priorStride = prior.strides_[levelPr];
            for (unsigned i=0; i<imax; ++i)
            {
                sum += marginalizeInnerLoop(idx, levelPr+1U, idxPr,
                                            prior, indexMap);
                idx += myStride;
                idxPr += priorStride;
            }
        }
        return sum;
    }

    template <typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,StackLen,StackDim>::marginalizeLoop(
        const unsigned level, unsigned long idx,
        const unsigned levelRes, unsigned long idxRes,
        const ArrayND<Num2,Len2,Dim2>& prior,
        const unsigned* indexMap, ArrayND& result) const
    {
        if (level == dim_)
        {
            const Numeric res = marginalizeInnerLoop(
                idx, 0U, 0UL, prior, indexMap);
            if (result.dim_)
                result.data_[idxRes] = res;
            else
                result.localData_[0] = res;
        }
        else
        {
            // Check if this level is mapped or not
            bool mapped = false;
            for (unsigned i=0; i<prior.dim_; ++i)
                if (level == indexMap[i])
                {
                    mapped = true;
                    break;
                }
            if (mapped)
                marginalizeLoop(level+1U, idx, levelRes, idxRes,
                                prior, indexMap, result);
            else
            {
                const unsigned imax = shape_[level];
                const unsigned long myStride = strides_[level];
                const unsigned long resStride = result.strides_[levelRes];
                for (unsigned i=0; i<imax; ++i)
                {
                    marginalizeLoop(level+1U, idx, levelRes+1U, idxRes,
                                    prior, indexMap, result);
                    idx += myStride;
                    idxRes += resStride;
                }
            }
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,StackLen,StackDim>
    ArrayND<Numeric,StackLen,StackDim>::marginalize(
        const ArrayND<Num2,Len2,Dim2>& prior,
        const unsigned* indexMap, const unsigned mapLen) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"marginalize\"");
        if (!(prior.dim_ && prior.dim_ <= dim_)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::marginalize: incompatible argument array rank");
        const unsigned resultDim = dim_ - prior.dim_;

        // Check that the index map is reasonable
        if (mapLen != prior.dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::marginalize: incompatible index map length");
        assert(indexMap);
        for (unsigned i=0; i<mapLen; ++i)
        {
            const unsigned thisInd = indexMap[i];
            if (shape_[thisInd] != prior.shape_[i]) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::marginalize: "
                "incompatible argument array dimensions");
            if (thisInd >= dim_) throw npstat::NpstatOutOfRange(
                "In npstat::ArrayND::marginalize: index map entry out of range");
            for (unsigned j=0; j<i; ++j)
                if (indexMap[j] == thisInd) throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::marginalize: "
                    "duplicate entry in the index map");
        }

        // Build the shape for the array of results
        ArrayShape newShape;
        newShape.reserve(resultDim);
        for (unsigned i=0; i<dim_; ++i)
        {
            bool mapped = false;
            for (unsigned j=0; j<mapLen; ++j)
                if (indexMap[j] == i)
                {
                    mapped = true;
                    break;
                }
            if (!mapped)
                newShape.push_back(shape_[i]);
        }

        ArrayND result(newShape);
        assert(result.dim_ == resultDim);
        bool calculated = false;
        if (resultDim == 0)
        {
            calculated = true;
            for (unsigned i=0; i<dim_; ++i)
                if (indexMap[i] != i)
                {
                    calculated = false;
                    break;
                }
            if (calculated)
            {
                Numeric sum = Numeric();
                for (unsigned long i=0; i<len_; ++i)
                    sum += data_[i]*prior.data_[i];
                result.localData_[0] = sum;
            }
        }

        if (!calculated)
            marginalizeLoop(0U, 0UL, 0U, 0UL, prior, indexMap, result);

        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    void ArrayND<Numeric,Len,Dim>::buildFromShapePtr(
        const unsigned* sizes, const unsigned dim)
    {
        dim_ = dim;
        if (dim_)
        {
            assert(sizes);
            for (unsigned i=0; i<dim_; ++i)
                if (sizes[i] == 0)
                    throw npstat::NpstatInvalidArgument(
                        "In npstat::ArrayND::buildFromShapePtr: "
                        "detected span of zero");

            // Copy the array shape and figure out the array length
            shape_ = makeBuffer(dim_, localShape_, Dim);
            for (unsigned i=0; i<dim_; ++i)
            {
                shape_[i] = sizes[i];
                len_ *= shape_[i];
            }

            // Figure out the array strides
            buildStrides();

            // Allocate the data array
            data_ = makeBuffer(len_, localData_, Len);
        }
        else
        {
            localData_[0] = Numeric();
            data_ = localData_;
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,StackLen,StackDim>::ArrayND(
        const ArrayND<Num2, Len2, Dim2>& slicedArray,
        const unsigned *fixedIndices, const unsigned nFixedIndices)
        : data_(0), strides_(0), shape_(0),
          len_(1UL), dim_(slicedArray.dim_ - nFixedIndices),
          shapeIsKnown_(true)
    {
        if (nFixedIndices)
        {
            assert(fixedIndices);
            if (nFixedIndices > slicedArray.dim_) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND slicing constructor: too many fixed indices");
            if (!slicedArray.shapeIsKnown_) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND slicing constructor: "
                "uninitialized argument array");

            // Check that the fixed indices are within range
            for (unsigned j=0; j<nFixedIndices; ++j)
                if (fixedIndices[j] >= slicedArray.dim_)
                    throw npstat::NpstatOutOfRange("In npstat::ArrayND slicing "
                        "constructor: fixed index out of range");

            // Build the shape for the slice
            shape_ = makeBuffer(dim_, localShape_, StackDim);
            unsigned idim = 0;
            for (unsigned i=0; i<slicedArray.dim_; ++i)
            {
                bool fixed = false;
                for (unsigned j=0; j<nFixedIndices; ++j)
                    if (fixedIndices[j] == i)
                    {
                        fixed = true;
                        break;
                    }
                if (!fixed)
                {
                    assert(idim < dim_);
                    shape_[idim++] = slicedArray.shape_[i];
                }
            }
            assert(idim == dim_);

            if (dim_)
            {
                // Copy the array shape and figure out the array length
                for (unsigned i=0; i<dim_; ++i)
                    len_ *= shape_[i];

                // Figure out the array strides
                buildStrides();

                // Allocate the data array
                data_ = makeBuffer(len_, localData_, StackLen);
            }
            else
            {
                localData_[0] = Numeric();
                data_ = localData_;
            }
        }
        else
        {
            new (this) ArrayND(slicedArray);
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    unsigned long ArrayND<Numeric,StackLen,StackDim>::verifySliceCompatibility(
        const ArrayND<Num2,Len2,Dim2>& slice,
        const unsigned *fixedIndices,
        const unsigned *fixedIndexValues,
        const unsigned nFixedIndices) const
    {
        if (!(nFixedIndices && nFixedIndices <= dim_))
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::verifySliceCompatibility: "
                "invalid number of fixed indices");
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling "
            "method \"verifySliceCompatibility\"");
        if (!slice.shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::verifySliceCompatibility: "
            "uninitialized argument array");
        if (slice.dim_ != dim_ - nFixedIndices) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::verifySliceCompatibility: "
            "incompatible argument array rank");
        assert(fixedIndices);
        assert(fixedIndexValues);

        for (unsigned j=0; j<nFixedIndices; ++j)
            if (fixedIndices[j] >= dim_) throw npstat::NpstatOutOfRange(
                "In npstat::ArrayND::verifySliceCompatibility: "
                "fixed index out of range");

        // Check slice shape compatibility
        unsigned long idx = 0UL;
        unsigned sliceDim = 0U;
        for (unsigned i=0; i<dim_; ++i)
        {
            bool fixed = false;
            for (unsigned j=0; j<nFixedIndices; ++j)
                if (fixedIndices[j] == i)
                {
                    fixed = true;
                    if (fixedIndexValues[j] >= shape_[i])
                        throw npstat::NpstatOutOfRange(
                            "In npstat::ArrayND::verifySliceCompatibility: "
                            "fixed index value out of range");
                    idx += fixedIndexValues[j]*strides_[i];
                    break;
                }
            if (!fixed)
            {
                if (shape_[i] != slice.shape_[sliceDim])
                     throw npstat::NpstatInvalidArgument(
                         "In npstat::ArrayND::verifySliceCompatibility: "
                         "incompatible argument array dimensions");
                ++sliceDim;
            }
        }       
        assert(sliceDim == slice.dim_);
        return idx;
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Op>
    void ArrayND<Numeric,StackLen,StackDim>::jointSliceLoop(
        const unsigned level, const unsigned long idx0,
        const unsigned level1, const unsigned long idx1,
        ArrayND<Num2,Len2,Dim2>& slice,
        const unsigned *fixedIndices,
        const unsigned *fixedIndexValues,
        const unsigned nFixedIndices,
        Op fcn)
    {
        bool fixed = false;
        for (unsigned j=0; j<nFixedIndices; ++j)
            if (fixedIndices[j] == level)
            {
                fixed = true;
                break;
            }
        if (fixed)
        {
            jointSliceLoop(level+1, idx0, level1, idx1,
                           slice, fixedIndices, fixedIndexValues,
                           nFixedIndices, fcn);
        }
        else
        {
            const unsigned imax = shape_[level];
            assert(imax == slice.shape_[level1]);
            const unsigned long stride = strides_[level];

            if (level1 == slice.dim_ - 1)
            {
                Num2* to = slice.data_ + idx1;
                for (unsigned i = 0; i<imax; ++i)
                    fcn(data_[idx0 + i*stride], to[i]);
            }
            else
            {
                const unsigned long stride2 = slice.strides_[level1];
                for (unsigned i = 0; i<imax; ++i)
                    jointSliceLoop(level+1, idx0+i*stride,
                                   level1+1, idx1+i*stride2,
                                   slice, fixedIndices, fixedIndexValues,
                                   nFixedIndices, fcn);
            }
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,StackLen,StackDim>::jointSliceScan(
        ArrayND<Num2,Len2,Dim2>& slice,
        const unsigned *fixedIndices,
        const unsigned *fixedIndexValues,
        const unsigned nFixedIndices,
        Functor binaryFunct)
    {
        const unsigned long idx = verifySliceCompatibility(
            slice, fixedIndices, fixedIndexValues, nFixedIndices);
        if (slice.dim_)
            jointSliceLoop(0U, idx, 0U, 0UL, slice, fixedIndices,
                           fixedIndexValues, nFixedIndices, binaryFunct);
        else
            binaryFunct(data_[idx], slice.localData_[0]);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2>
    void ArrayND<Numeric,StackLen,StackDim>::projectInnerLoop(
        const unsigned level, unsigned long idx0,
        unsigned* currentIndex,
        AbsArrayProjector<Numeric,Num2>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        // level :  dimension number among indices which are being iterated
        const unsigned idx = projectedIndices[level];
        const unsigned imax = shape_[idx];
        const unsigned long stride = strides_[idx];
        const bool last = (level == nProjectedIndices - 1);

        for (unsigned i = 0; i<imax; ++i)
        {
            currentIndex[idx] = i;
            if (last)
                projector.process(currentIndex, dim_, idx0, data_[idx0]);
            else
                projectInnerLoop(level+1, idx0, currentIndex, projector, 
                                 projectedIndices, nProjectedIndices);
            idx0 += stride;
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2,
              typename Num3, class Op>
    void ArrayND<Numeric,StackLen,StackDim>::projectLoop(
        const unsigned level, const unsigned long idx0,
        const unsigned level1, const unsigned long idx1,
        unsigned* currentIndex,
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsArrayProjector<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices, Op fcn) const
    {
        // level        : dimension number in this array
        // level1       : dimension number in the projection
        // idx0         : linear index in this array
        // idx1         : linear index in the projection
        // currentIndex : cycled over in this loop with the exception of the
        //                dimensions which are iterated over to build the
        //                projection
        if (level == dim_)
        {
            assert(level1 == projection->dim_);
            projector.clear();
            projectInnerLoop(0U, idx0, currentIndex, projector,
                             projectedIndices, nProjectedIndices);
            if (projection->dim_)
                fcn(projection->data_[idx1], projector.result());
            else
                fcn(projection->localData_[0], projector.result());
        }
        else
        {
            bool iterated = false;
            for (unsigned j=0; j<nProjectedIndices; ++j)
                if (projectedIndices[j] == level)
                {
                    iterated = true;
                    break;
                }
            if (iterated)
            {
                // This index will be iterated over inside "projectInnerLoop"
                projectLoop(level+1, idx0, level1, idx1,
                            currentIndex, projection, projector,
                            projectedIndices, nProjectedIndices, fcn);
            }
            else
            {
                const unsigned imax = shape_[level];
                const unsigned long stride = strides_[level];
                // We will not be able to get here if projection->dim_ is 0.
                // Therefore, it is safe to access projection->strides_.
                const unsigned long stride2 = projection->strides_[level1];
                for (unsigned i = 0; i<imax; ++i)
                {
                    currentIndex[level] = i;
                    projectLoop(level+1, idx0+i*stride,
                                level1+1, idx1+i*stride2,
                                currentIndex, projection, projector,
                                projectedIndices, nProjectedIndices, fcn);
                }
            }
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,StackLen,StackDim>::verifyProjectionCompatibility(
        const ArrayND<Num2,Len2,Dim2>& projection,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        if (!(nProjectedIndices && nProjectedIndices <= dim_))
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::verifyProjectionCompatibility: "
                "invalid number of projected indices");
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling "
            "method \"verifyProjectionCompatibility\"");
        if (!projection.shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::verifyProjectionCompatibility: "
            "uninitialized argument array");
        if (projection.dim_ != dim_ - nProjectedIndices)
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::verifyProjectionCompatibility: "
                "incompatible argument array rank");
        assert(projectedIndices);

        for (unsigned j=0; j<nProjectedIndices; ++j)
            if (projectedIndices[j] >= dim_) throw npstat::NpstatOutOfRange(
                "In npstat::ArrayND::verifyProjectionCompatibility: "
                "projected index out of range");

        // Check projection shape compatibility
        unsigned sliceDim = 0U;
        for (unsigned i=0; i<dim_; ++i)
        {
            bool fixed = false;
            for (unsigned j=0; j<nProjectedIndices; ++j)
                if (projectedIndices[j] == i)
                {
                    fixed = true;
                    break;
                }
            if (!fixed)
            {
                if (shape_[i] != projection.shape_[sliceDim])
                     throw npstat::NpstatInvalidArgument(
                         "In npstat::ArrayND::verifyProjectionCompatibility: "
                         "incompatible argument array dimensions");
                ++sliceDim;
            }
        }
        assert(sliceDim == projection.dim_);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
    void ArrayND<Numeric,StackLen,StackDim>::project(
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsArrayProjector<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        verifyProjectionCompatibility(*projection, projectedIndices,
                                      nProjectedIndices);
        unsigned ibuf[StackDim];
        unsigned* buf = makeBuffer(dim_, ibuf, StackDim);
        for (unsigned i=0; i<dim_; ++i)
            buf[i] = 0U;
        projectLoop(0U, 0UL, 0U, 0UL, buf, projection,
                    projector, projectedIndices, nProjectedIndices,
                    scast_assign_left<Num2,Num3>());
        destroyBuffer(buf, ibuf);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
    void ArrayND<Numeric,StackLen,StackDim>::addToProjection(
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsArrayProjector<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        verifyProjectionCompatibility(*projection, projectedIndices,
                                      nProjectedIndices);
        unsigned ibuf[StackDim];
        unsigned* buf = makeBuffer(dim_, ibuf, StackDim);
        for (unsigned i=0; i<dim_; ++i)
            buf[i] = 0U;
        projectLoop(0U, 0UL, 0U, 0UL, buf, projection,
                    projector, projectedIndices, nProjectedIndices,
                    scast_pluseq_left<Num2,Num3>());        
        destroyBuffer(buf, ibuf);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
    void ArrayND<Numeric,StackLen,StackDim>::subtractFromProjection(
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsArrayProjector<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        verifyProjectionCompatibility(*projection, projectedIndices,
                                      nProjectedIndices);
        unsigned ibuf[StackDim];
        unsigned* buf = makeBuffer(dim_, ibuf, StackDim);
        for (unsigned i=0; i<dim_; ++i)
            buf[i] = 0U;
        projectLoop(0U, 0UL, 0U, 0UL, buf, projection,
                    projector, projectedIndices, nProjectedIndices,
                    scast_minuseq_left<Num2,Num3>());        
        destroyBuffer(buf, ibuf);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2>
    void ArrayND<Numeric,StackLen,StackDim>::projectInnerLoop2(
        const unsigned level, const unsigned long idx0,
        AbsVisitor<Numeric,Num2>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        const unsigned idx = projectedIndices[level];
        const unsigned imax = shape_[idx];
        const unsigned long stride = strides_[idx];
        const bool last = (level == nProjectedIndices - 1);

        for (unsigned i = 0; i<imax; ++i)
        {
            if (last)
                projector.process(data_[idx0+i*stride]);
            else
                projectInnerLoop2(level+1, idx0+i*stride, projector, 
                                  projectedIndices, nProjectedIndices);
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2,
              typename Num3, class Op>
    void ArrayND<Numeric,StackLen,StackDim>::projectLoop2(
        const unsigned level, const unsigned long idx0,
        const unsigned level1, const unsigned long idx1,
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsVisitor<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices, Op fcn) const
    {
        if (level == dim_)
        {
            assert(level1 == projection->dim_);
            projector.clear();
            projectInnerLoop2(0U, idx0, projector,
                              projectedIndices, nProjectedIndices);
            if (projection->dim_)
                fcn(projection->data_[idx1], projector.result());
            else
                fcn(projection->localData_[0], projector.result());
        }
        else
        {
            bool fixed = false;
            for (unsigned j=0; j<nProjectedIndices; ++j)
                if (projectedIndices[j] == level)
                {
                    fixed = true;
                    break;
                }
            if (fixed)
            {
                projectLoop2(level+1, idx0, level1, idx1,
                             projection, projector,
                             projectedIndices, nProjectedIndices, fcn);
            }
            else
            {
                const unsigned imax = shape_[level];
                const unsigned long stride = strides_[level];
                const unsigned long stride2 = projection->strides_[level1];
                for (unsigned i = 0; i<imax; ++i)
                    projectLoop2(level+1, idx0+i*stride,
                                 level1+1, idx1+i*stride2,
                                 projection, projector,
                                 projectedIndices, nProjectedIndices, fcn);
            }
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
    void ArrayND<Numeric,StackLen,StackDim>::project(
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsVisitor<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        verifyProjectionCompatibility(*projection, projectedIndices,
                                      nProjectedIndices);
        projectLoop2(0U, 0UL, 0U, 0UL, projection,
                     projector, projectedIndices, nProjectedIndices,
                     scast_assign_left<Num2,Num3>());
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
    void ArrayND<Numeric,StackLen,StackDim>::addToProjection(
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsVisitor<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        verifyProjectionCompatibility(*projection, projectedIndices,
                                      nProjectedIndices);
        projectLoop2(0U, 0UL, 0U, 0UL, projection,
                     projector, projectedIndices, nProjectedIndices,
                     scast_pluseq_left<Num2,Num3>());
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, typename Num3>
    void ArrayND<Numeric,StackLen,StackDim>::subtractFromProjection(
        ArrayND<Num2,Len2,Dim2>* projection,
        AbsVisitor<Numeric,Num3>& projector,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices) const
    {
        assert(projection);
        verifyProjectionCompatibility(*projection, projectedIndices,
                                      nProjectedIndices);
        projectLoop2(0U, 0UL, 0U, 0UL, projection,
                     projector, projectedIndices, nProjectedIndices,
                     scast_minuseq_left<Num2,Num3>());
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, class Functor>
    void ArrayND<Numeric,StackLen,StackDim>::scaleBySliceInnerLoop(
        const unsigned level, const unsigned long idx0,
        Num2& scale, const unsigned *projectedIndices,
        const unsigned nProjectedIndices, Functor binaryFunct)
    {
        const unsigned idx = projectedIndices[level];
        const unsigned imax = shape_[idx];
        const unsigned long stride = strides_[idx];

        if (level == nProjectedIndices - 1)
        {
            Numeric* data = data_ + idx0;
            for (unsigned i = 0; i<imax; ++i)
                binaryFunct(data[i*stride], scale);
        }
        else
            for (unsigned i = 0; i<imax; ++i)
                scaleBySliceInnerLoop(level+1, idx0+i*stride, scale,
                                      projectedIndices, nProjectedIndices,
                                      binaryFunct);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,StackLen,StackDim>::scaleBySliceLoop(
        unsigned level, unsigned long idx0,
        unsigned level1, unsigned long idx1,
        ArrayND<Num2,Len2,Dim2>& slice,
        const unsigned *projectedIndices,
        const unsigned nProjectedIndices,
        Functor binaryFunct)
    {
        if (level == dim_)
        {
            assert(level1 == slice.dim_);
            Num2& scaleFactor = slice.dim_ ? slice.data_[idx1] : 
                                             slice.localData_[0];
            scaleBySliceInnerLoop(0U, idx0, scaleFactor, projectedIndices,
                                  nProjectedIndices, binaryFunct);
        }
        else
        {
            bool fixed = false;
            for (unsigned j=0; j<nProjectedIndices; ++j)
                if (projectedIndices[j] == level)
                {
                    fixed = true;
                    break;
                }
            if (fixed)
            {
                scaleBySliceLoop(level+1, idx0, level1, idx1, slice,
                                 projectedIndices, nProjectedIndices,
                                 binaryFunct);
            }
            else
            {
                const unsigned imax = shape_[level];
                const unsigned long stride = strides_[level];
                const unsigned long stride2 = slice.strides_[level1];
                for (unsigned i = 0; i<imax; ++i)
                    scaleBySliceLoop(level+1, idx0+i*stride, level1+1,
                                     idx1+i*stride2, slice, projectedIndices,
                                     nProjectedIndices, binaryFunct);
            }
        }
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,StackLen,StackDim>::jointScan(
        ArrayND<Num2, Len2, Dim2>& r, Functor binaryFunct)
    {
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::jointScan: incompatible argument array shape");
        if (dim_)
            for (unsigned long i=0; i<len_; ++i)
                binaryFunct(data_[i], r.data_[i]);
        else
            binaryFunct(localData_[0], r.localData_[0]);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,StackLen,StackDim>::applySlice(
        ArrayND<Num2,Len2,Dim2>& slice,
        const unsigned *fixedIndices, const unsigned nFixedIndices,
        Functor binaryFunct)
    {
        if (nFixedIndices)
        {
            verifyProjectionCompatibility(slice, fixedIndices, nFixedIndices);
            if (slice.dim_ == 0U)
                for (unsigned long i=0; i<len_; ++i)
                    binaryFunct(data_[i], slice.localData_[0]);
            else
                scaleBySliceLoop(0U, 0UL, 0U, 0UL, slice,
                                 fixedIndices, nFixedIndices, binaryFunct);
        }
        else
            jointScan(slice, binaryFunct);
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    inline bool ArrayND<Numeric,StackLen,StackDim>::isCompatible(
        const ArrayShape& shape) const
    {
        if (!shapeIsKnown_)
            return false;
        if (dim_ != shape.size())
            return false;
        if (dim_)
        {
            for (unsigned i=0; i<dim_; ++i)
                if (shape_[i] != shape[i])
                    return false;
        }
        else
            assert(len_ == 1UL);
        return true;
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    inline bool ArrayND<Numeric,StackLen,StackDim>::isShapeCompatible(
        const ArrayND<Num2,Len2,Dim2>& r) const
    {
        if (!shapeIsKnown_)
            return false;
        if (!r.shapeIsKnown_)
            return false;
        if (dim_ != r.dim_)
            return false;
        if (len_ != r.len_)
            return false;
        if (dim_)
        {
            assert(shape_);
            assert(r.shape_);
            for (unsigned i=0; i<dim_; ++i)
                if (shape_[i] != r.shape_[i])
                    return false;
        }
        else
            assert(len_ == 1UL);
        return true;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2, typename Integer>
    void ArrayND<Numeric,Len,Dim>::processSubrangeLoop(
        const unsigned level, unsigned long idx0,
        unsigned* currentIndex,
        AbsArrayProjector<Numeric,Num2>& f,
        const BoxND<Integer>& subrange) const
    {
        // Deal with possible negative limits first
        const Interval<Integer>& levelRange(subrange[level]);
        long long int iminl = static_cast<long long int>(levelRange.min());
        if (iminl < 0LL) iminl = 0LL;
        long long int imaxl = static_cast<long long int>(levelRange.max());
        if (imaxl < 0LL) imaxl = 0LL;

        // Now deal with possible out-of-range limits
        const unsigned imin = static_cast<unsigned>(iminl);
        unsigned imax = static_cast<unsigned>(imaxl);
        if (imax > shape_[level])
            imax = shape_[level];

        if (level == dim_ - 1)
        {
            idx0 += imin;
            for (unsigned i=imin; i<imax; ++i, ++idx0)
            {
                currentIndex[level] = i;
                f.process(currentIndex, dim_, idx0, data_[idx0]);
            }
        }
        else
        {
            const unsigned long stride = strides_[level];
            idx0 += imin*stride;
            for (unsigned i=imin; i<imax; ++i)
            {
                currentIndex[level] = i;
                processSubrangeLoop(level+1U, idx0, currentIndex, f, subrange);
                idx0 += stride;
            }
        }
    }

    template<typename Numeric, unsigned Len, unsigned StackDim>
    template <typename Num2, typename Integer>
    void ArrayND<Numeric,Len,StackDim>::processSubrange(
        AbsArrayProjector<Numeric,Num2>& f,
        const BoxND<Integer>& subrange) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"processSubrange\"");
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::processSubrange method "
            "can not be used with array of 0 rank");
        if (dim_ != subrange.dim()) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::processSubrange: incompatible subrange rank");
        unsigned ibuf[StackDim];
        unsigned* buf = makeBuffer(dim_, ibuf, StackDim);
        for (unsigned i=0; i<dim_; ++i)
            buf[i] = 0U;
        processSubrangeLoop(0U, 0UL, buf, f, subrange);
        destroyBuffer(buf, ibuf);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2>
    inline ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::setData(
        const Num2* data, const unsigned long dataLength)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"setData\"");
        if (dataLength != len_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::setData: incompatible input data length");
        copyBuffer(data_, data, dataLength);
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    void ArrayND<Numeric,Len,Dim>::buildStrides()
    {
        assert(dim_);
        if (strides_ == 0)
            strides_ = makeBuffer(dim_, localStrides_, Dim);
        strides_[dim_ - 1] = 1UL;
        for (unsigned j=dim_ - 1; j>0; --j)
            strides_[j - 1] = strides_[j]*shape_[j];
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    inline ArrayND<Numeric,StackLen,StackDim>::ArrayND()
        : data_(0), strides_(0), shape_(0),
          len_(0UL), dim_(0U), shapeIsKnown_(false)
    {
        localData_[0] = Numeric();
        data_ = localData_;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const ArrayND& r)
        : data_(0), strides_(0), shape_(0),
          len_(r.len_), dim_(r.dim_), shapeIsKnown_(r.shapeIsKnown_)
    {
        if (dim_)
        {
            // Copy the shape
            shape_ = makeBuffer(dim_, localShape_, Dim);
            copyBuffer(shape_, r.shape_, dim_);

            // Copy the strides
            strides_ = makeBuffer(dim_, localStrides_, Dim);
            copyBuffer(strides_, r.strides_, dim_);

            // Copy the data
            data_ = makeBuffer(len_, localData_, Len);
            copyBuffer(data_, r.data_, len_);
        }
        else
        {
            assert(len_ == 1UL);
            localData_[0] = r.localData_[0];
            data_ = localData_;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>::ArrayND(const ArrayND<Num2, Len2, Dim2>& r)
        : data_(0), strides_(0), shape_(0),
          len_(r.len_), dim_(r.dim_), shapeIsKnown_(r.shapeIsKnown_)
    {
        if (dim_)
        {
            // Copy the shape
            shape_ = makeBuffer(dim_, localShape_, Dim);
            copyBuffer(shape_, r.shape_, dim_);

            // Copy the strides
            strides_ = makeBuffer(dim_, localStrides_, Dim);
            copyBuffer(strides_, r.strides_, dim_);

            // Copy the data
            data_ = makeBuffer(len_, localData_, Len);
            copyBuffer(data_, r.data_, len_);
        }
        else
        {
            assert(len_ == 1UL);
            localData_[0] = static_cast<Numeric>(r.localData_[0]);
            data_ = localData_;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    ArrayND<Numeric,Len,Dim>::ArrayND(const ArrayND<Num2, Len2, Dim2>& r,
                                      Functor f)
        : data_(0), strides_(0), shape_(0),
          len_(r.len_), dim_(r.dim_), shapeIsKnown_(r.shapeIsKnown_)
    {
        if (dim_)
        {
            // Copy the shape
            shape_ = makeBuffer(dim_, localShape_, Dim);
            copyBuffer(shape_, r.shape_, dim_);

            // Copy the strides
            strides_ = makeBuffer(dim_, localStrides_, Dim);
            copyBuffer(strides_, r.strides_, dim_);

            // Copy the data
            data_ = makeBuffer(len_, localData_, Len);
            for (unsigned long i=0; i<len_; ++i)
                data_[i] = static_cast<Numeric>(f(r.data_[i]));
        }
        else
        {
            assert(len_ == 1UL);
            localData_[0] = static_cast<Numeric>(f(r.localData_[0]));
            data_ = localData_;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    void ArrayND<Numeric,Len,Dim>::copyRangeLoopFunct(
        const unsigned level, unsigned long idx0,
        unsigned long idx1,
        const ArrayND<Num2, Len2, Dim2>& r,
        const ArrayRange& range, Functor f)
    {
        const unsigned imax = shape_[level];
        if (level == dim_ - 1)
        {
            Numeric* to = data_ + idx0;
            const Num2* from = r.data_ + (idx1 + range[level].min());
            for (unsigned i=0; i<imax; ++i)
                *to++ = static_cast<Numeric>(f(*from++));
        }
        else
        {
            const unsigned long fromstride = r.strides_[level];
            const unsigned long tostride = strides_[level];
            idx1 += range[level].min()*fromstride;
            for (unsigned i=0; i<imax; ++i)
            {
                copyRangeLoopFunct(level+1, idx0, idx1, r, range, f);
                idx0 += tostride;
                idx1 += fromstride;
            }
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>::ArrayND(
        const ArrayND<Num2, Len2, Dim2>& r, const ArrayRange& range)
        : data_(0), strides_(0), shape_(0),
          len_(r.len_), dim_(r.dim_), shapeIsKnown_(r.shapeIsKnown_)
    {
        if (!range.isCompatible(r.shape_, r.dim_))
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND subrange constructor: invalid subrange");
        if (dim_)
        {
            len_ = range.rangeSize();
            if (!len_)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND subrange constructor: empty subrange");

            // Figure out the shape
            shape_ = makeBuffer(dim_, localShape_, Dim);
            range.rangeLength(shape_, dim_);

            // Figure out the strides
            buildStrides();

            // Allocate the data array
            data_ = makeBuffer(len_, localData_, Len);

            // Copy the data
            if (dim_ > CHAR_BIT*sizeof(unsigned long))
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND subrange constructor: "
                    "input array rank is too large");
            unsigned lolim[CHAR_BIT*sizeof(unsigned long)];
            range.lowerLimits(lolim, dim_);
            unsigned toBuf[CHAR_BIT*sizeof(unsigned long)];
            clearBuffer(toBuf, dim_);
            (const_cast<ArrayND<Num2, Len2, Dim2>&>(r)).commonSubrangeLoop(
                0U, 0UL, 0UL, lolim, shape_, toBuf, *this,
                scast_assign_right<Num2,Numeric>());
        }
        else
        {
            assert(len_ == 1UL);
            localData_[0] = static_cast<Numeric>(r.localData_[0]);
            data_ = localData_;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    ArrayND<Numeric,Len,Dim>::ArrayND(
        const ArrayND<Num2, Len2, Dim2>& r, const ArrayRange& range,
        Functor f)
        : data_(0), strides_(0), shape_(0),
          len_(r.len_), dim_(r.dim_), shapeIsKnown_(r.shapeIsKnown_)
    {
        if (!range.isCompatible(r.shape_, r.dim_))
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND transforming subrange constructor: "
                "incompatible subrange");
        if (dim_)
        {
            len_ = range.rangeSize();
            if (!len_)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND transforming subrange constructor: "
                    "empty subrange");

            // Figure out the shape
            shape_ = makeBuffer(dim_, localShape_, Dim);
            for (unsigned i=0; i<dim_; ++i)
                shape_[i] = range[i].length();

            // Figure out the strides
            buildStrides();

            // Allocate the data array
            data_ = makeBuffer(len_, localData_, Len);

            // Transform the data
            copyRangeLoopFunct(0U, 0UL, 0UL, r, range, f);
        }
        else
        {
            assert(len_ == 1UL);
            localData_[0] = static_cast<Numeric>(f(r.localData_[0]));
            data_ = localData_;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const ArrayShape& sh)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned sz = sh.size();
        buildFromShapePtr(sz ? &sh[0] : 0, sz);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned* sizes,
                                      const unsigned dim)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 1U;
        unsigned sizes[dim];
        sizes[0] = n0;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 2U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 3U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2,
                                      const unsigned n3)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 4U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        sizes[3] = n3;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2,
                                      const unsigned n3,
                                      const unsigned n4)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 5U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        sizes[3] = n3;
        sizes[4] = n4;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2,
                                      const unsigned n3,
                                      const unsigned n4,
                                      const unsigned n5)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 6U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        sizes[3] = n3;
        sizes[4] = n4;
        sizes[5] = n5;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2,
                                      const unsigned n3,
                                      const unsigned n4,
                                      const unsigned n5,
                                      const unsigned n6)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 7U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        sizes[3] = n3;
        sizes[4] = n4;
        sizes[5] = n5;
        sizes[6] = n6;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2,
                                      const unsigned n3,
                                      const unsigned n4,
                                      const unsigned n5,
                                      const unsigned n6,
                                      const unsigned n7)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 8U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        sizes[3] = n3;
        sizes[4] = n4;
        sizes[5] = n5;
        sizes[6] = n6;
        sizes[7] = n7;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2,
                                      const unsigned n3,
                                      const unsigned n4,
                                      const unsigned n5,
                                      const unsigned n6,
                                      const unsigned n7,
                                      const unsigned n8)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 9U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        sizes[3] = n3;
        sizes[4] = n4;
        sizes[5] = n5;
        sizes[6] = n6;
        sizes[7] = n7;
        sizes[8] = n8;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>::ArrayND(const unsigned n0,
                                      const unsigned n1,
                                      const unsigned n2,
                                      const unsigned n3,
                                      const unsigned n4,
                                      const unsigned n5,
                                      const unsigned n6,
                                      const unsigned n7,
                                      const unsigned n8,
                                      const unsigned n9)
        : data_(0), strides_(0), shape_(0), len_(1UL), shapeIsKnown_(true)
    {
        const unsigned dim = 10U;
        unsigned sizes[dim];
        sizes[0] = n0;
        sizes[1] = n1;
        sizes[2] = n2;
        sizes[3] = n3;
        sizes[4] = n4;
        sizes[5] = n5;
        sizes[6] = n6;
        sizes[7] = n7;
        sizes[8] = n8;
        sizes[9] = n9;
        buildFromShapePtr(sizes, dim);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num1, unsigned Len1, unsigned Dim1,
             typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,Len,Dim>::outerProductLoop(
        const unsigned level, unsigned long idx0,
        unsigned long idx1, unsigned long idx2,
        const ArrayND<Num1, Len1, Dim1>& a1,
        const ArrayND<Num2, Len2, Dim2>& a2)
    {
        const unsigned imax = shape_[level];
        if (level == dim_ - 1)
        {
            for (unsigned i=0; i<imax; ++i)
                data_[idx0 + i] = a1.data_[idx1]*a2.data_[idx2 + i];
        }
        else
        {
            for (unsigned i=0; i<imax; ++i)
            {
                outerProductLoop(level+1, idx0, idx1, idx2, a1, a2);
                idx0 += strides_[level];
                if (level < a1.dim_)
                    idx1 += a1.strides_[level];
                else
                    idx2 += a2.strides_[level - a1.dim_];
            }                                 
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num1, unsigned Len1, unsigned Dim1,
             typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>::ArrayND(const ArrayND<Num1, Len1, Dim1>& a1,
                                      const ArrayND<Num2, Len2, Dim2>& a2)
        : data_(0), strides_(0), shape_(0),
          len_(1UL), dim_(a1.dim_ + a2.dim_), shapeIsKnown_(true)
    {
        if (!(a1.shapeIsKnown_ && a2.shapeIsKnown_))
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND outer product constructor: "
                "uninitialized argument array");
        if (dim_)
        {
            shape_ = makeBuffer(dim_, localShape_, Dim);
            copyBuffer(shape_, a1.shape_, a1.dim_);
            copyBuffer(shape_+a1.dim_, a2.shape_, a2.dim_);

            for (unsigned i=0; i<dim_; ++i)
            {
                assert(shape_[i]);
                len_ *= shape_[i];
            }

            // Figure out the array strides
            buildStrides();

            // Allocate the data array
            data_ = makeBuffer(len_, localData_, Len);

            // Fill the data array
            if (a1.dim_ == 0)
            {
                for (unsigned long i=0; i<len_; ++i)
                    data_[i] = a1.localData_[0] * a2.data_[i];
            }
            else if (a2.dim_ == 0)
            {
                for (unsigned long i=0; i<len_; ++i)
                    data_[i] = a1.data_[i] * a2.localData_[0];
            }
            else
                outerProductLoop(0U, 0UL, 0UL, 0UL, a1, a2);
        }
        else
        {
            localData_[0] = a1.localData_[0] * a2.localData_[0];
            data_ = localData_;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline ArrayND<Numeric,Len,Dim>::~ArrayND()
    {
        destroyBuffer(data_, localData_);
        destroyBuffer(strides_, localStrides_);
        destroyBuffer(shape_, localShape_);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::operator=(const ArrayND& r)
    {
        if (this == &r)
            return *this;
        if (shapeIsKnown_)
        {
            if (!r.shapeIsKnown_) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND assignment operator: "
                "uninitialized argument array");
            if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND assignment operator: "
                "incompatible argument array shape");
            if (dim_)
                copyBuffer(data_, r.data_, len_);
            else
                localData_[0] = r.localData_[0];
        }
        else
        {
            // This object is uninitialized. If the object on the
            // right is itself initialized, make an in-place copy.
            if (r.shapeIsKnown_)
                new (this) ArrayND(r);
        }
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::operator=(const ArrayND<Num2,Len2,Dim2>& r)
    {
        if ((void*)this == (void*)(&r))
            return *this;
        if (shapeIsKnown_)
        {
            if (!r.shapeIsKnown_) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND assignment operator: "
                "uninitialized argument array");
            if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND assignment operator: "
                "incompatible argument array shape");
            if (dim_)
                copyBuffer(data_, r.data_, len_);
            else
                localData_[0] = static_cast<Numeric>(r.localData_[0]);
        }
        else
        {
            // This object is uninitialized. If the object on the
            // right is itself initialized, make an in-place copy.
            if (r.shapeIsKnown_)
                new (this) ArrayND(r);
        }
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2, class Functor>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::assign(const ArrayND<Num2,Len2,Dim2>& r,
                                     Functor f)
    {
        if (shapeIsKnown_)
        {
            if (!r.shapeIsKnown_) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::assign: uninitialized argument array");
            if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::assign: incompatible argument array shape");
            if (dim_)
                for (unsigned long i=0; i<len_; ++i)
                    data_[i] = static_cast<Numeric>(f(r.data_[i]));
            else
                localData_[0] = static_cast<Numeric>(f(r.localData_[0]));
        }
        else
        {
            // This object is uninitialized. If the object on the
            // right is itself initialized, build new array in place.
            if (r.shapeIsKnown_)
                new (this) ArrayND(r, f);
        }
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline ArrayShape ArrayND<Numeric,Len,Dim>::shape() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"shape\"");
        return ArrayShape(shape_, shape_+dim_);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline ArrayRange ArrayND<Numeric,Len,Dim>::fullRange() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"fullRange\"");
        ArrayRange range;
        if (dim_)
        {
            range.reserve(dim_);
            for (unsigned i=0; i<dim_; ++i)
                range.push_back(Interval<unsigned>(0U, shape_[i]));
        }
        return range;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    bool ArrayND<Numeric,Len,Dim>::isDensity() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"isDensity\"");
        const Numeric zero = Numeric();
        bool hasPositive = false;
        if (dim_)
            for (unsigned long i=0; i<len_; ++i)
            {
                // Don't make comparisons whose result can be
                // determined in advance by assuming that Numeric
                // is an unsigned type. Some compilers will
                // complain about it when this template is
                // instantiated with such a type.
                if (data_[i] == zero)
                    continue;
                if (ComplexComparesFalse<Numeric>::less(zero, data_[i]))
                    hasPositive = true;
                else
                    return false;
            }
        else
            hasPositive = ComplexComparesFalse<Numeric>::less(
                zero, localData_[0]);
        return hasPositive;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    bool ArrayND<Numeric,Len,Dim>::isZero() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"isZero\"");
        const Numeric zero = Numeric();
        if (dim_)
        {
            for (unsigned long i=0; i<len_; ++i)
                if (data_[i] != zero)
                    return false;
        }
        else
            if (localData_[0] != zero)
                return false;
        return true;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    void ArrayND<Numeric,Len,Dim>::convertLinearIndex(
        unsigned long l, unsigned* idx, const unsigned idxLen) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling "
            "method \"convertLinearIndex\"");
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::convertLinearIndex method "
            "can not be used with array of 0 rank");
        if (idxLen != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::convertLinearIndex: incompatible index length");
        if (l >= len_) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::convertLinearIndex: linear index out of range");
        assert(idx);

        for (unsigned i=0; i<dim_; ++i)
        {
            idx[i] = l / strides_[i];
            l -= (idx[i] * strides_[i]);
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    unsigned long ArrayND<Numeric,Len,Dim>::linearIndex(
        const unsigned* index, unsigned idxLen) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"linearIndex\"");
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::linearIndex method "
            "can not be used with array of 0 rank");
        if (idxLen != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::linearIndex: incompatible index length");
        assert(index);

        unsigned long idx = 0UL;
        for (unsigned i=0; i<dim_; ++i)
        {
            if (index[i] >= shape_[i])
                throw npstat::NpstatOutOfRange(
                    "In npstat::ArrayND::linearIndex: index out of range");
            idx += index[i]*strides_[i];
        }
        return idx;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::value(
        const unsigned *index, const unsigned dim)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"value\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::value: incompatible index length");
        if (dim)
        {
            assert(index);
            unsigned long idx = 0UL;
            for (unsigned i=0; i<dim_; ++i)
                idx += index[i]*strides_[i];
            return data_[idx];
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::value(
        const unsigned *index, const unsigned dim) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"value\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::value: incompatible index length");
        if (dim)
        {
            assert(index);
            unsigned long idx = 0UL;
            for (unsigned i=0; i<dim_; ++i)
                idx += index[i]*strides_[i];
            return data_[idx];
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::linearValue(
        const unsigned long index)
    {
        return data_[index];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::linearValue(
        const unsigned long index) const
    {
        return data_[index];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::linearValueAt(
        const unsigned long index)
    {
        if (index >= len_)
            throw npstat::NpstatOutOfRange(
                "In npstat::ArrayND::linearValueAt: linear index out of range");
        return data_[index];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::linearValueAt(
        const unsigned long index) const
    {
        if (index >= len_)
            throw npstat::NpstatOutOfRange(
                "In npstat::ArrayND::linearValueAt: linear index out of range");
        return data_[index];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline unsigned ArrayND<Numeric,Len,Dim>::coordToIndex(
        const double x, const unsigned idim) const
    {
        if (x <= 0.0)
            return 0;
        else if (x >= static_cast<double>(shape_[idim] - 1))
            return shape_[idim] - 1;
        else
            return static_cast<unsigned>(std::floor(x + 0.5));
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::closest(
        const double *x, const unsigned dim) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"closest\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::closest: incompatible data length");
        if (dim)
        {
            assert(x);
            unsigned long idx = 0UL;
            for (unsigned i=0; i<dim_; ++i)
                idx += coordToIndex(x[i], i)*strides_[i];
            return data_[idx];
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::closest(
        const double *x, const unsigned dim)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"closest\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::closest: incompatible data length");
        if (dim)
        {
            assert(x);
            unsigned long idx = 0UL;
            for (unsigned i=0; i<dim_; ++i)
                idx += coordToIndex(x[i], i)*strides_[i];
            return data_[idx];
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::valueAt(
        const unsigned *index, const unsigned dim) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"valueAt\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::valueAt: incompatible index length");
        if (dim)
        {
            assert(index);
            unsigned long idx = 0UL;
            for (unsigned i=0; i<dim_; ++i)
            {
                if (index[i] >= shape_[i]) throw npstat::NpstatOutOfRange(
                    "In npstat::ArrayND::valueAt: index out of range");
                idx += index[i]*strides_[i];
            }
            return data_[idx];
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::valueAt(
        const unsigned *index, const unsigned dim)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"valueAt\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::valueAt: incompatible index length");
        if (dim)
        {
            assert(index);
            unsigned long idx = 0UL;
            for (unsigned i=0; i<dim_; ++i)
            {
                if (index[i] >= shape_[i]) throw npstat::NpstatOutOfRange(
                    "In npstat::ArrayND::valueAt: index out of range");
                idx += index[i]*strides_[i];
            }
            return data_[idx];
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()()
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator()\"");
        if (dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 0 array)");
        return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator()\"");
        if (dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 0 array)");
        return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i)
    {
        if (1U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 1 array)");
        return data_[i];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i) const
    {
        if (1U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 1 array)");
        return data_[i];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"at\"");
        if (dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 0 array)");
        return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at()
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"at\"");
        if (dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 0 array)");
        return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0) const
    {
        if (1U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 1 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 1)");
        return data_[i0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0)
    {
        if (1U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 1 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 1)");
        return data_[i0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1)
    {
        if (2U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 2 array)");
        return data_[i0*strides_[0] + i1];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1) const
    {
        if (2U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 2 array)");
        return data_[i0*strides_[0] + i1];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1) const
    {
        if (2U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 2 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 2)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 2)");
        return data_[i0*strides_[0] + i1];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1)
    {
        if (2U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 2 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 2)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 2)");
        return data_[i0*strides_[0] + i1];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2) const 
    {
        if (3U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 3 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3) const 
    {
        if (4U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 4 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + i3];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4) const 
    {
        if (5U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 5 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5) const 
    {
        if (6U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 6 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6) const 
    {
        if (7U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 7 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] + i6];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7) const 
    {
        if (8U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 8 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8) const 
    {
        if (9U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 9 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8,
        const unsigned i9) const 
    {
        if (10U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 10 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8*strides_[8] + i9];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2)
    {
        if (3U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 3 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3)
    {
        if (4U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 4 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + i3];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4)
    {
        if (5U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 5 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5)
    {
        if (6U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 6 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6)
    {
        if (7U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 7 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] + i6];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7)
    {
        if (8U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 8 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8)
    {
        if (9U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 9 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::operator()(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8,
        const unsigned i9)
    {
        if (10U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator(): wrong # of args (not rank 10 array)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8*strides_[8] + i9];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2) const 
    {
        if (3U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 3 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 3)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 3)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 3)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3) const 
    {
        if (4U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 4 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 4)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 4)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 4)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 4)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + i3];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4) const 
    {
        if (5U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 5 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 5)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 5)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 5)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 5)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 5)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5) const 
    {
        if (6U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 6 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 6)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 6)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 6)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 6)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 6)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 6)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6) const 
    {
        if (7U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 7 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 7)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 7)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 7)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 7)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 7)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 7)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 7)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] + i6];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7) const 
    {
        if (8U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 8 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 8)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 8)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 8)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 8)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 8)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 8)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 8)");
        if (i7 >= shape_[7]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 7 out of range (rank 8)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8) const 
    {
        if (9U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 9 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 9)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 9)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 9)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 9)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 9)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 9)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 9)");
        if (i7 >= shape_[7]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 7 out of range (rank 9)");
        if (i8 >= shape_[8]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 8 out of range (rank 9)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    const Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8,
        const unsigned i9) const 
    {
        if (10U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 10 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 10)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 10)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 10)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 10)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 10)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 10)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 10)");
        if (i7 >= shape_[7]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 7 out of range (rank 10)");
        if (i8 >= shape_[8]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 8 out of range (rank 10)");
        if (i9 >= shape_[9]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 9 out of range (rank 10)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8*strides_[8] + i9];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2)
    {
        if (3U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 3 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 3)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 3)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 3)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3)
    {
        if (4U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 4 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 4)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 4)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 4)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 4)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + i3];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4)
    {
        if (5U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 5 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 5)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 5)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 5)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 5)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 5)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5)
    {
        if (6U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 6 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 6)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 6)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 6)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 6)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 6)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 6)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6)
    {
        if (7U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 7 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 7)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 7)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 7)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 7)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 7)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 7)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 7)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] + i6];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7)
    {
        if (8U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 8 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 8)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 8)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 8)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 8)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 8)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 8)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 8)");
        if (i7 >= shape_[7]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 7 out of range (rank 8)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8)
    {
        if (9U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 9 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 9)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 9)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 9)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 9)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 9)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 9)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 9)");
        if (i7 >= shape_[7]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 7 out of range (rank 9)");
        if (i8 >= shape_[8]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 8 out of range (rank 9)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric& ArrayND<Numeric,Len,Dim>::at(
        const unsigned i0,
        const unsigned i1,
        const unsigned i2,
        const unsigned i3,
        const unsigned i4,
        const unsigned i5,
        const unsigned i6,
        const unsigned i7,
        const unsigned i8,
        const unsigned i9)
    {
        if (10U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::at: wrong # of args (not rank 10 array)");
        if (i0 >= shape_[0]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 0 out of range (rank 10)");
        if (i1 >= shape_[1]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 1 out of range (rank 10)");
        if (i2 >= shape_[2]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 2 out of range (rank 10)");
        if (i3 >= shape_[3]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 3 out of range (rank 10)");
        if (i4 >= shape_[4]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 4 out of range (rank 10)");
        if (i5 >= shape_[5]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 5 out of range (rank 10)");
        if (i6 >= shape_[6]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 6 out of range (rank 10)");
        if (i7 >= shape_[7]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 7 out of range (rank 10)");
        if (i8 >= shape_[8]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 8 out of range (rank 10)");
        if (i9 >= shape_[9]) throw npstat::NpstatOutOfRange(
            "In npstat::ArrayND::at: index 9 out of range (rank 10)");
        return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + 
                     i3*strides_[3] + i4*strides_[4] + i5*strides_[5] +
                     i6*strides_[6] + i7*strides_[7] + i8*strides_[8] + i9];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<unsigned Len2, unsigned Dim2>
    double ArrayND<Numeric,Len,Dim>::maxAbsDifference(
        const ArrayND<Numeric,Len2,Dim2>& r) const
    {
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::maxAbsDifference: "
            "incompatible argument array shape");
        if (dim_)
        {
            double maxd = 0.0;
            for (unsigned long i=0; i<len_; ++i)
            {
                const Numeric rval = r.data_[i];
                const double d = absDifference(data_[i], rval);
                if (d > maxd)
                    maxd = d;
            }
            return maxd;
        }
        else
        {
            const Numeric rval = r.localData_[0];
            return absDifference(localData_[0], rval);
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<unsigned Len2, unsigned Dim2>
    bool ArrayND<Numeric,Len,Dim>::operator==(
        const ArrayND<Numeric,Len2,Dim2>& r) const
    {
        if (shapeIsKnown_ != r.shapeIsKnown_)
            return false;
        if (r.dim_ != dim_)
            return false;
        if (r.len_ != len_)
            return false;
        for (unsigned i=0; i<dim_; ++i)
            if (shape_[i] != r.shape_[i])
                return false;
        for (unsigned i=0; i<dim_; ++i)
            assert(strides_[i] == r.strides_[i]);
        for (unsigned long j=0; j<len_; ++j)
            if (data_[j] != r.data_[j])
                return false;
        return true;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<unsigned Len2, unsigned Dim2>
    inline bool ArrayND<Numeric,Len,Dim>::operator!=(
        const ArrayND<Numeric,Len2,Dim2>& r) const
    {
        return !(*this == r);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2>
    ArrayND<Numeric,Len,Dim>
    ArrayND<Numeric,Len,Dim>::operator*(const Num2& r) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator*\"");
        ArrayND<Numeric,Len,Dim> result(shape_, dim_);
        for (unsigned long i=0; i<len_; ++i)
            result.data_[i] = data_[i]*r;
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2>
    ArrayND<Numeric,Len,Dim>
    ArrayND<Numeric,Len,Dim>::operator/(const Num2& r) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator/\"");
        if (r == Num2()) throw npstat::NpstatRuntimeError(
            "In npstat::ArrayND::operator/: division by zero");
        ArrayND<Numeric,Len,Dim> result(shape_, dim_);
        for (unsigned long i=0; i<len_; ++i)
            result.data_[i] = data_[i]/r;
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>
    ArrayND<Numeric,Len,Dim>::operator+(
        const ArrayND<Numeric,Len2,Dim2>& r) const
    {
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator+: "
            "incompatible argument array shape");
        ArrayND<Numeric,Len,Dim> result(shape_, dim_);
        for (unsigned long i=0; i<len_; ++i)
            result.data_[i] = data_[i] + r.data_[i];
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>
    ArrayND<Numeric,Len,Dim>::operator-(
        const ArrayND<Numeric,Len2,Dim2>& r) const
    {
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator-: "
            "incompatible argument array shape");
        ArrayND<Numeric,Len,Dim> result(shape_, dim_);
        for (unsigned long i=0; i<len_; ++i)
            result.data_[i] = data_[i] - r.data_[i];
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::operator+() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator+\"");
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::operator-() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator-\"");
        ArrayND<Numeric,Len,Dim> result(shape_, dim_);
        for (unsigned long i=0; i<len_; ++i)
            result.data_[i] = -data_[i];
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::operator*=(const Num2& r)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator*=\"");
        for (unsigned long i=0; i<len_; ++i)
            data_[i] *= r;
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::makeNonNegative()
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"makeNonNegative\"");
        const Numeric zero = Numeric();
        if (dim_)
        {
            for (unsigned long i=0; i<len_; ++i)
                if (!(ComplexComparesAbs<Numeric>::more(data_[i], zero)))
                    data_[i] = zero;
        }
        else
            if (!(ComplexComparesAbs<Numeric>::more(localData_[0], zero)))
                localData_[0] = zero;
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    unsigned ArrayND<Numeric,Len,Dim>::makeCopulaSteps(
        const double tolerance, const unsigned nCycles)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"makeCopulaSteps\"");
        if (nCycles == 0U)
            return 0U;
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::makeCopulaSteps method "
            "can not be used with array of 0 rank");

        const Numeric zero = Numeric();
        for (unsigned long i=0; i<len_; ++i)
            if (!(ComplexComparesAbs<Numeric>::more(data_[i], zero)))
                data_[i] = zero;

        std::vector<Numeric*> axesPtrBuf(dim_);
        Numeric** axes = &axesPtrBuf[0];
        const Numeric one = static_cast<Numeric>(1);

        // Memory for the axis accumulators
        unsigned idxSum = 0;
        for (unsigned i=0; i<dim_; ++i)
            idxSum += shape_[i];
        std::vector<Numeric> axesBuf(idxSum);
        axes[0] = &axesBuf[0];
        for (unsigned i=1; i<dim_; ++i)
            axes[i] = axes[i-1] + shape_[i-1];

        // Accumulate axis projections
        unsigned icycle = 0;
        for (; icycle<nCycles; ++icycle)
        {
            for (unsigned i=0; i<idxSum; ++i)
                axesBuf[i] = zero;

            // Accumulate sums for each axis
            for (unsigned long idat=0; idat<len_; ++idat)
            {
                unsigned long l = idat;
                for (unsigned i=0; i<dim_; ++i)
                {
                    const unsigned idx = l / strides_[i];
                    l -= (idx * strides_[i]);
                    axes[i][idx] += data_[idat];
                }
            }

            // Make averages out of sums
            bool withinTolerance = true;
            Numeric totalSum = zero;
            for (unsigned i=0; i<dim_; ++i)
            {
                Numeric axisSum = zero;
                const unsigned amax = shape_[i];
                for (unsigned a=0; a<amax; ++a)
                {
                    if (axes[i][a] == zero)
                        throw npstat::NpstatRuntimeError(
                            "In npstat::ArrayND::makeCopulaSteps: "
                            "marginal density is zero");
                    axisSum += axes[i][a];
                }
                totalSum += axisSum;
                const Numeric axisAverage = axisSum/static_cast<Numeric>(amax);
                for (unsigned a=0; a<amax; ++a)
                    axes[i][a] /= axisAverage;
                for (unsigned a=0; a<amax && withinTolerance; ++a)
                {
                    const double adelta = absDifference(axes[i][a], one);
                    if (adelta > tolerance)
                        withinTolerance = false;
                }
            }

            if (withinTolerance)
                break;

            const Numeric totalAverage = totalSum/
                static_cast<Numeric>(len_)/static_cast<Numeric>(dim_);

            // Run over all points again and divide by
            // the product of marginals
            for (unsigned long idat=0; idat<len_; ++idat)
            {
                unsigned long l = idat;
                for (unsigned i=0; i<dim_; ++i)
                {
                    const unsigned idx = l / strides_[i];
                    l -= (idx * strides_[i]);
                    data_[idat] /= axes[i][idx];
                }
                data_[idat] /= totalAverage;
            }
        }

        return icycle;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::operator/=(const Num2& r)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"operator/=\"");
        if (r == Num2()) throw npstat::NpstatRuntimeError(
            "In npstat::ArrayND::operator/=: division by zero");
        for (unsigned long i=0; i<len_; ++i)
            data_[i] /= r;
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::operator+=(const ArrayND<Num2,Len2,Dim2>& r)
    {
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator+=: "
            "incompatible argument array shape");
        for (unsigned long i=0; i<len_; ++i)
            data_[i] += r.data_[i];
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num3, typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::addmul(const ArrayND<Num2,Len2,Dim2>& r,
                                     const Num3& c)
    {
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::addmul: "
            "incompatible argument array shape");
        for (unsigned long i=0; i<len_; ++i)
            data_[i] += r.data_[i]*c;
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim>&
    ArrayND<Numeric,Len,Dim>::operator-=(const ArrayND<Num2,Len2,Dim2>& r)
    {
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::operator-=: "
            "incompatible argument array shape");
        for (unsigned long i=0; i<len_; ++i)
            data_[i] -= r.data_[i];
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric ArrayND<Numeric,Len,Dim>::interpolate1(
        const double *coords, const unsigned dim) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"interpolate1\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::interpolate1: incompatible coordinate length");
        if (dim)
        {
            const unsigned maxdim = CHAR_BIT*sizeof(unsigned long);
            if (dim_ >= maxdim)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::interpolate1: array rank is too large");

            double dx[maxdim];
            unsigned ix[maxdim];
            for (unsigned i=0; i<dim; ++i)
            {
                const double x = coords[i];
                if (x <= 0.0)
                {
                    ix[i] = 0;
                    dx[i] = 0.0;
                }
                else if (x >= static_cast<double>(shape_[i] - 1))
                {
                    ix[i] = shape_[i] - 1;
                    dx[i] = 0.0;
                }
                else
                {
                    ix[i] = static_cast<unsigned>(std::floor(x));
                    dx[i] = x - ix[i];
                }
            }

            Numeric sum = Numeric();
            const unsigned long maxcycle = 1UL << dim;
            for (unsigned long icycle=0UL; icycle<maxcycle; ++icycle)
            {
                double w = 1.0;
                unsigned long icell = 0UL;
                for (unsigned i=0; i<dim; ++i)
                {
                    if (icycle & (1UL << i))
                    {
                        w *= dx[i];
                        icell += strides_[i]*(ix[i] + 1U);
                    }
                    else
                    {
                        w *= (1.0 - dx[i]);
                        icell += strides_[i]*ix[i];
                    }
                }
                if (w > 0.0)
                    sum += data_[icell]*static_cast<proper_double>(w);
            }
            return sum;
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric ArrayND<Numeric,Len,Dim>::interpolateLoop(
        const unsigned level, const double* coords, const Numeric* base) const
    {
        const unsigned npoints = shape_[level];
        const double x = coords[level];

        unsigned ix, npt = 1;
        double dx = 0.0;
        if (x < 0.0)
            ix = 0;
        else if (x > static_cast<double>(npoints - 1))
            ix = npoints - 1;
        else
        {
            ix = static_cast<unsigned>(std::floor(x));
            if (ix) --ix;
            unsigned imax = ix + 3;
            while (imax >= npoints)
            {
                if (ix) --ix;
                --imax;
            }
            dx = x - ix;
            npt = imax + 1 - ix;
        }
        assert(npt >= 1 && npt <= 4);

        Numeric fit[4];
        if (level < dim_ - 1)
            for (unsigned ipt=0; ipt<npt; ++ipt)
                fit[ipt] = interpolateLoop(level + 1, coords,
                                           base + (ix + ipt)*strides_[level]);

        const Numeric* const v = (level == dim_ - 1 ? base + ix : fit);
        switch (npt)
        {
        case 1:
            return v[0];
        case 2:
            return interpolate_linear(dx, v[0], v[1]);
        case 3:
            return interpolate_quadratic(dx, v[0], v[1], v[2]);
        case 4:
            return interpolate_cubic(dx, v[0], v[1], v[2], v[3]);
        default:
            assert(0);
            return Numeric();
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric ArrayND<Numeric,Len,Dim>::interpolate3(
        const double* coords, const unsigned dim) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"interpolate3\"");
        if (dim != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::interpolate3: incompatible coordinate length");
        if (dim)
        {
            assert(coords);
            return interpolateLoop(0, coords, data_);
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<class Functor>
    ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::apply(Functor f)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"apply\"");
        for (unsigned long i=0; i<len_; ++i)
            data_[i] = static_cast<Numeric>(f(data_[i]));
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<class Functor>
    ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::scanInPlace(
        Functor f)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"scanInPlace\"");
        for (unsigned long i=0; i<len_; ++i)
            f(data_[i]);
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::constFill(
        const Numeric c)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"constFill\"");
        for (unsigned long i=0; i<len_; ++i)
            data_[i] = c;
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::clear()
    {
        return constFill(Numeric());
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::uninitialize()
    {
        destroyBuffer(data_, localData_);
        destroyBuffer(strides_, localStrides_);
        destroyBuffer(shape_, localShape_);
        localData_[0] = Numeric();
        data_ = localData_;
        strides_ = 0;
        shape_ = 0;
        len_ = 0;
        dim_ = 0;
        shapeIsKnown_ = false;
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::makeUnit()
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"makeUnit\"");
        if (dim_ < 2) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::makeUnit method "
            "can not be used with arrays of rank less than 2");
        constFill(Numeric());
        unsigned long stride = 0UL;
        const unsigned dimlen = shape_[0];
        for (unsigned i=0; i<dim_; ++i)
        {
            if (shape_[i] != dimlen) throw npstat::NpstatInvalidArgument(
                "npstat::ArrayND::makeUnit method needs "
                "the array span to be the same in ech dimension");
            stride += strides_[i];
        }
        const Numeric one(static_cast<Numeric>(1));
        for (unsigned i=0; i<dimlen; ++i)
            data_[i*stride] = one;
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    void ArrayND<Numeric,Len,Dim>::linearFillLoop(
        const unsigned level, const double s0, const unsigned long idx,
        const double shift, const double* coeffs)
    {
        const unsigned imax = shape_[level];
        const double c = coeffs[level];
        if (level == dim_ - 1)
        {
            Numeric* d = &data_[idx];
            for (unsigned i=0; i<imax; ++i)
            {
                // Note that we want to add "shift" only at the
                // very end. This might improve the numerical
                // precision of the result.
                const double sum = s0 + c*i + shift;
                d[i] = static_cast<Numeric>(sum);
            }
        }
        else
        {
            const unsigned long stride = strides_[level];
            for (unsigned i=0; i<imax; ++i)
                linearFillLoop(level+1, s0 + c*i, idx + i*stride,
                               shift, coeffs);
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::linearFill(
        const double* coeffs, const unsigned dimCoeffs, const double shift)
    {
        // Make sure the object has been initialized
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"linearFill\"");
        if (dim_ != dimCoeffs) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::linearFill: incompatible number of coefficients");
        if (dim_)
        {
            assert(coeffs);
            linearFillLoop(0U, 0.0, 0UL, shift, coeffs);
        }
        else
            localData_[0] = static_cast<Numeric>(shift);
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<class Functor>
    void ArrayND<Numeric,Len,Dim>::functorFillLoop(
        const unsigned level, const unsigned long idx,
        Functor f, unsigned* farg)
    {
        const unsigned imax = shape_[level];
        if (level == dim_ - 1)
        {
            Numeric* d = &data_[idx];
            const unsigned* myarg = farg;
            for (unsigned i = 0; i<imax; ++i)
            {
                farg[level] = i;
                d[i] = static_cast<Numeric>(f(myarg, dim_));
            }
        }
        else
        {
            const unsigned long stride = strides_[level];
            for (unsigned i = 0; i<imax; ++i)
            {
                farg[level] = i;
                functorFillLoop(level+1, idx + i*stride, f, farg);
            }
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Accumulator>
    void ArrayND<Numeric,Len,Dim>::convertToLastDimCdfLoop(
        ArrayND* sumSlice, const unsigned level, unsigned long idx0,
        unsigned long idxSlice, const bool useTrapezoids)
    {
        static const proper_double half = 0.5;
        const unsigned imax = shape_[level];
        if (level == dim_ - 1)
        {
            Accumulator acc = Accumulator();
            Numeric* data = data_ + idx0;
            if (useTrapezoids)
            {
                Numeric oldval = Numeric();
                for (unsigned i = 0; i<imax; ++i)
                {
                    acc += (data[i] + oldval)*half;
                    oldval = data[i];
                    data[i] = static_cast<Numeric>(acc);
                }
                acc += oldval*half;
            }
            else
                for (unsigned i = 0; i<imax; ++i)
                {
                    acc += data[i];
                    data[i] = static_cast<Numeric>(acc);
                }
            if (sumSlice->dim_)
                sumSlice->data_[idxSlice] = static_cast<Numeric>(acc);
            else
                sumSlice->localData_[0] = static_cast<Numeric>(acc);
        }
        else
        {
            const unsigned long stride = strides_[level];
            unsigned long sumStride = 0UL;
            if (sumSlice->dim_)
                sumStride = sumSlice->strides_[level];
            for (unsigned i = 0; i<imax; ++i)
            {
                convertToLastDimCdfLoop<Accumulator>(
                    sumSlice, level+1, idx0, idxSlice, useTrapezoids);
                idx0 += stride;
                idxSlice += sumStride;
            }
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Accumulator>
    inline void ArrayND<Numeric,Len,Dim>::convertToLastDimCdf(
        ArrayND* sumSlice, const bool useTrapezoids)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling "
            "method \"convertToLastDimCdf\"");
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::convertToLastDimCdf method "
            "can not be used with array of 0 rank");
        assert(sumSlice);
        if (!sumSlice->shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::convertToLastDimCdf: "
            "uninitialized argument array");
        convertToLastDimCdfLoop<Accumulator>(sumSlice, 0U, 0UL, 0UL,
                                             useTrapezoids);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<class Functor>
    ArrayND<Numeric,Len,Dim>& ArrayND<Numeric,Len,Dim>::functorFill(Functor f)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"functorFill\"");
        if (dim_)
        {
            unsigned localIndex[Dim];
            unsigned* index = makeBuffer(dim_, localIndex, Dim);
            functorFillLoop(0U, 0UL, f, index);
            destroyBuffer(index, localIndex);
        }
        else
            localData_[0] = static_cast<Numeric>(
                f(static_cast<unsigned*>(0), 0U));
        return *this;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    bool ArrayND<Numeric,Len,Dim>::isClose(
        const ArrayND<Num2,Len2,Dim2>& r, const double eps) const
    {
        if (eps < 0.0) throw npstat::NpstatDomainError(
            "In npstat::ArrayND::isClose: tolerance must not be negative");
        if (!isShapeCompatible(r)) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::isClose: incompatible argument array shape");
        if (dim_)
        {
            for (unsigned long i=0; i<len_; ++i)
            {
                const Numeric rval = r.data_[i];
                if (static_cast<double>(absDifference(data_[i], rval)) > eps)
                    return false;
            }
        }
        else
        {
            const Numeric rval = r.localData_[0];
            if (static_cast<double>(absDifference(localData_[0], rval)) > eps)
                return false;
        }
        return true;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::outer(
        const ArrayND<Num2,Len2,Dim2>& r) const
    {
        return ArrayND<Numeric,Len,Dim>(*this, r);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    void ArrayND<Numeric,Len,Dim>::contractLoop(
        unsigned thisLevel, const unsigned resLevel,
        const unsigned pos1, const unsigned pos2,
        unsigned long idxThis, unsigned long idxRes, ArrayND& result) const
    {
        while (thisLevel == pos1 || thisLevel == pos2)
            ++thisLevel;
        assert(thisLevel < dim_);

        if (resLevel == result.dim_ - 1)
        {
            const unsigned ncontract = shape_[pos1];
            const unsigned imax = result.shape_[resLevel];
            const unsigned long stride = strides_[pos1] + strides_[pos2];
            for (unsigned i=0; i<imax; ++i)
            {
                const Numeric* tmp = data_ + (idxThis + i*strides_[thisLevel]);
                Numeric sum = Numeric();
                for (unsigned j=0; j<ncontract; ++j)
                    sum += tmp[j*stride];
                result.data_[idxRes + i] = sum;
            }
        }
        else
        {
            const unsigned imax = result.shape_[resLevel];
            assert(imax == shape_[thisLevel]);
            for (unsigned i=0; i<imax; ++i)
            {
                contractLoop(thisLevel+1, resLevel+1, pos1, pos2,
                             idxThis, idxRes, result);
                idxThis += strides_[thisLevel];
                idxRes += result.strides_[resLevel];
            }
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::contract(
        const unsigned pos1, const unsigned pos2) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"contract\"");
        if (!(pos1 < dim_ && pos2 < dim_ && pos1 != pos2))
            throw npstat::NpstatInvalidArgument("In npstat::ArrayND::contract: "
                                        "incompatible contraction indices");
        if (shape_[pos1] != shape_[pos2])
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::contract: incompatible "
                "length of contracted dimensions");

        // Construct the new shape
        unsigned newshapeBuf[Dim];
        unsigned* newshape = makeBuffer(dim_ - 2, newshapeBuf, Dim);
        unsigned ishap = 0;
        for (unsigned i=0; i<dim_; ++i)
            if (i != pos1 && i != pos2)
                newshape[ishap++] = shape_[i];

        // Form the result array
        ArrayND<Numeric,Len,Dim> result(newshape, ishap);
        if (ishap)
            contractLoop(0, 0, pos1, pos2, 0UL, 0UL, result);
        else
        {
            // We are just calculating the trace
            Numeric sum = Numeric();
            const unsigned imax = shape_[0];
            const unsigned long stride = strides_[0] + strides_[1];
            for (unsigned i=0; i<imax; ++i)
                sum += data_[i*stride];
            result() = sum;
        }

        destroyBuffer(newshape, newshapeBuf);
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    void ArrayND<Numeric,Len,Dim>::transposeLoop(
        const unsigned level, const unsigned pos1, const unsigned pos2,
        unsigned long idxThis, unsigned long idxRes, ArrayND& result) const
    {
        const unsigned imax = shape_[level];
        const unsigned long mystride = strides_[level];
        const unsigned relevel = level == pos1 ? pos2 : 
            (level == pos2 ? pos1 : level);
        const unsigned long restride = result.strides_[relevel];
        const bool ready = (level == dim_ - 1);
        for (unsigned i=0; i<imax; ++i)
        {
            if (ready)
                result.data_[idxRes] = data_[idxThis];
            else
                transposeLoop(level+1, pos1, pos2, idxThis, idxRes, result);
            idxThis += mystride;
            idxRes += restride;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2>
    Num2 ArrayND<Numeric,Len,Dim>::sum() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"sum\"");
        Num2 sum = Num2();
        for (unsigned long i=0; i<len_; ++i)
            sum += data_[i];
        return sum;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2>
    Num2 ArrayND<Numeric,Len,Dim>::sumsq() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"sumsq\"");
        Num2 sum = Num2();
        for (unsigned long i=0; i<len_; ++i)
        {
            const Num2 absval = absValue(data_[i]);
            sum += absval*absval;
        }
        return sum;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric ArrayND<Numeric,Len,Dim>::min() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"min\"");
        if (dim_)
        {
            Numeric minval(data_[0]);
            for (unsigned long i=1UL; i<len_; ++i)
                if (ComplexComparesAbs<Numeric>::less(data_[i], minval))
                    minval = data_[i];
            return minval;
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric ArrayND<Numeric,Len,Dim>::min(
        unsigned *index, const unsigned indexLen) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"min\"");
        if (indexLen != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::min: incompatible index length");
        if (dim_)
        {
            unsigned long minind = 0UL;
            Numeric minval(data_[0]);
            for (unsigned long i=1UL; i<len_; ++i)
                if (ComplexComparesAbs<Numeric>::less(data_[i], minval))
                {
                    minval = data_[i];
                    minind = i;
                }
            convertLinearIndex(minind, index, indexLen);
            return minval;
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric ArrayND<Numeric,Len,Dim>::max() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"max\"");
        if (dim_)
        {
            Numeric maxval(data_[0]);
            for (unsigned long i=1UL; i<len_; ++i)
                if (ComplexComparesAbs<Numeric>::less(maxval, data_[i]))
                    maxval = data_[i];
            return maxval;
        }
        else
            return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    Numeric ArrayND<Numeric,Len,Dim>::max(
        unsigned *index, const unsigned indexLen) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"max\"");
        if (indexLen != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::max: incompatible index length");
        if (dim_)
        {
            unsigned long maxind = 0UL;
            Numeric maxval(data_[0]);
            for (unsigned long i=1UL; i<len_; ++i)
                if (ComplexComparesAbs<Numeric>::less(maxval, data_[i]))
                {
                    maxval = data_[i];
                    maxind = i;
                }
            convertLinearIndex(maxind, index, indexLen);
            return maxval;
        }
        else
            return localData_[0];
    }

    // Faster function for 2d transpose
    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::transpose() const
    {
        if (dim_ != 2) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::transpose method "
            "can not be used with arrays of rank other than 2");
        unsigned newshape[2];
        newshape[0] = shape_[1];
        newshape[1] = shape_[0];
        ArrayND<Numeric,Len,Dim> result(newshape, dim_);
        const unsigned imax = shape_[0];
        const unsigned jmax = shape_[1];
        for (unsigned i=0; i<imax; ++i)
            for (unsigned j=0; j<jmax; ++j)
                result.data_[j*imax + i] = data_[i*jmax + j];
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Accumulator>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::derivative(
        const double inscale) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"derivative\"");
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::derivative method "
            "can not be used with array of 0 rank");

        const typename ProperDblFromCmpl<Accumulator>::type scale = inscale;
        const unsigned maxdim = CHAR_BIT*sizeof(unsigned long);
        if (dim_ >= maxdim) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::derivative: array rank is too large");
        const unsigned long maxcycle = 1UL << dim_;

        ArrayShape sh;
        sh.reserve(dim_);
        for (unsigned i=0; i<dim_; ++i)
        {
            if (shape_[i] <= 1U)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::derivative: in some dimendions "
                    "array size is too small");
            sh.push_back(shape_[i] - 1U);
        }

        ArrayND result(sh);
        const unsigned long rLen = result.length();
        for (unsigned long ilin=0; ilin<rLen; ++ilin)
        {
            result.convertLinearIndex(ilin, &sh[0], dim_);

            Accumulator deriv = Accumulator();
            for (unsigned long icycle=0UL; icycle<maxcycle; ++icycle)
            {
                unsigned long icell = 0UL;
                unsigned n1 = 0U;
                for (unsigned i=0; i<dim_; ++i)
                {
                    if (icycle & (1UL << i))
                    {
                        ++n1;
                        icell += strides_[i]*(sh[i] + 1);
                    }
                    else
                        icell += strides_[i]*sh[i];
                }
                if ((dim_ - n1) % 2U)
                    deriv -= data_[icell];
                else
                    deriv += data_[icell];
            }
            result.data_[ilin] = static_cast<Numeric>(deriv*scale);
        }

        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Accumulator>
    Accumulator ArrayND<Numeric,Len,Dim>::sumBelowLoop(
        const unsigned level, unsigned long idx0,
        const unsigned* limit) const
    {
        Accumulator cdf = Accumulator();
        const unsigned imax = limit[level] + 1U;
        if (level == dim_ - 1)
        {
            Numeric* base = data_ + idx0;
            for (unsigned i=0; i<imax; ++i)
                cdf += base[i];
        }
        else
        {
            const unsigned long stride = strides_[level];
            for (unsigned i=0; i<imax; ++i, idx0+=stride)
                cdf += sumBelowLoop<Accumulator>(level+1, idx0, limit);
        }
        return cdf;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Accumulator>
    Accumulator ArrayND<Numeric,Len,Dim>::cdfValue(
        const unsigned *index, const unsigned indexLen) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"cdfValue\"");
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::cdfValue method "
            "can not be used with array of 0 rank");
        if (indexLen != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cdfValue: incompatible index length");
        for (unsigned i=0; i<indexLen; ++i)
            if (index[i] >= shape_[i])
                throw npstat::NpstatOutOfRange(
                    "In npstat::ArrayND::cdfValue: index out of range");
        return sumBelowLoop<Accumulator>(0, 0U, index);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template <typename Accumulator>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::cdfArray(
        const double inscale) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"cdfArray\"");
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::cdfArray method "
            "can not be used with array of 0 rank");

        const proper_double scale = inscale;
        const unsigned maxdim = CHAR_BIT*sizeof(unsigned long);
        if (dim_ >= maxdim)
            throw npstat::NpstatInvalidArgument(
                "In npstat::ArrayND::cdfArray: array rank is too large");
        const unsigned long maxcycle = 1UL << dim_;

        ArrayShape sh;
        sh.reserve(dim_);
        for (unsigned i=0; i<dim_; ++i)
            sh.push_back(shape_[i] + 1U);

        ArrayND<Accumulator> result(sh);

        unsigned* psh = &sh[0];
        const unsigned long len = result.length();
        for (unsigned long ipre=0; ipre<len; ++ipre)
        {
            result.convertLinearIndex(ipre, psh, dim_);
            Accumulator deriv = Accumulator();
            bool has0 = false;
            for (unsigned i=0; i<dim_; ++i)
                if (psh[i]-- == 0U)
                {
                    has0 = true;
                    break;
                }
            if (!has0)
            {
                for (unsigned long icycle=0UL; icycle<maxcycle; ++icycle)
                {
                    unsigned long icell = 0UL;
                    unsigned n1 = 0U;
                    for (unsigned i=0; i<dim_; ++i)
                    {
                        if (icycle & (1UL << i))
                        {
                            ++n1;
                            icell += result.strides_[i]*(psh[i] + 1);
                        }
                        else
                            icell += result.strides_[i]*psh[i];
                    }
                    if (n1 < dim_)
                    {
                        if ((dim_ - n1) % 2U)
                            deriv += result.data_[icell];
                        else
                            deriv -= result.data_[icell];
                    }
                }
                deriv += static_cast<Accumulator>(value(psh, dim_)*scale);
            }
            result.data_[ipre] = deriv;
        }

        // The "return" will convert Accumulator type into Numeric
        return result;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::transpose(
        const unsigned pos1, const unsigned pos2) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"transpose\"");
        if (!(pos1 < dim_ && pos2 < dim_ && pos1 != pos2))
            throw npstat::NpstatInvalidArgument("In npstat::ArrayND::transpose: "
                                        "incompatible transposition indices");
        if (dim_ == 2)
            return transpose();
        else
        {
            // Construct the new shape
            unsigned newshapeBuf[Dim];
            unsigned *newshape = makeBuffer(dim_, newshapeBuf, Dim);
            copyBuffer(newshape, shape_, dim_);
            std::swap(newshape[pos1], newshape[pos2]);

            // Form the result array
            ArrayND<Numeric,Len,Dim> result(newshape, dim_);

            // Fill the result array
            transposeLoop(0, pos1, pos2, 0UL, 0UL, result);

            destroyBuffer(newshape, newshapeBuf);
            return result;
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,Len,Dim>::multiMirror(
        ArrayND<Num2, Len2, Dim2>* out) const
    {
        assert(out);
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"multiMirror\"");
        if (!out->shapeIsKnown_)
            *out = ArrayND<Num2, Len2, Dim2>(doubleShape(shape()));
        if (dim_ != out->dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::multiMirror: incompatible argument array rank");

        if (dim_)
        {
            const unsigned *dshape = out->shape_;
            for (unsigned i=0; i<dim_; ++i)
                if (dshape[i] != shape_[i]*2U) throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::multiMirror: "
                    "incompatible argument array shape");

            if (dim_ >= CHAR_BIT*sizeof(unsigned long))
                 throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::multiMirror: "
                    "array rank is too large");
            const unsigned long maxcycle = 1UL << dim_;
            std::vector<unsigned> indexbuf(dim_*2U);
            unsigned* idx = &indexbuf[0];
            unsigned* mirror = idx + dim_;

            for (unsigned long ipt=0; ipt<len_; ++ipt)
            {
                unsigned long l = ipt;
                for (unsigned i=0; i<dim_; ++i)
                {
                    idx[i] = l / strides_[i];
                    l -= (idx[i] * strides_[i]);
                }
                for (unsigned long icycle=0UL; icycle<maxcycle; ++icycle)
                {
                    for (unsigned i=0; i<dim_; ++i)
                    {
                        if (icycle & (1UL << i))
                            mirror[i] = dshape[i] - idx[i] - 1U;
                        else
                            mirror[i] = idx[i];
                    }
                    out->value(mirror, dim_) = data_[ipt];
                }
            }
        }
        else
            out->localData_[0] = static_cast<Num2>(localData_[0]);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,Len,Dim>::rotate(
        const unsigned* shifts, const unsigned lenShifts,
        ArrayND<Num2, Len2, Dim2>* rotated) const
    {
        assert(rotated);
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"rotate\"");
        // Can't rotate into itself -- it will be a mess
        if ((void*)rotated == (void*)this) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::rotate: can not rotate array into itself");
        if (!rotated->shapeIsKnown_)
            *rotated = *this;
        if (dim_ != rotated->dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::rotate: incompatible argument array rank");
        if (lenShifts != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::rotate: incompatible dimensionality of shifts");

        if (dim_)
        {
            assert(shifts);
            if (dim_ > CHAR_BIT*sizeof(unsigned long))
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::rotate: array rank is too large");
            unsigned buf[CHAR_BIT*sizeof(unsigned long)];
            clearBuffer(buf, dim_);
            (const_cast<ArrayND*>(this))->flatCircularLoop(
                0U, 0UL, 0UL, buf, shape_, shifts,
                *rotated, scast_assign_right<Numeric,Num2>());
        }
        else
            rotated->localData_[0] = static_cast<Num2>(localData_[0]);
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,Len,Dim>::dotProductLoop(
        const unsigned level, unsigned long idx0,
        unsigned long idx1, unsigned long idx2,
        const ArrayND<Num2, Len2, Dim2>& r,
        ArrayND& result) const
    {
        // idx0 -- this object
        // idx1 -- dot product argument
        // idx2 -- result
        if (level == result.dim_)
        {
            Numeric sum = Numeric();
            const unsigned imax = r.shape_[0];
            const unsigned rstride = r.strides_[0];
            const Numeric* l = data_ + idx0;
            const Num2* ri = r.data_ + idx1;
            for (unsigned i=0; i<imax; ++i)
                sum += l[i]*ri[i*rstride];
            result.data_[idx2] = sum;
        }
        else
        {
            const unsigned imax = result.shape_[level];
            for (unsigned i=0; i<imax; ++i)
            {
                dotProductLoop(level+1, idx0, idx1, idx2, r, result);
                idx2 += result.strides_[level];
                if (level < dim_ - 1)
                    idx0 += strides_[level];
                else
                    idx1 += r.strides_[level + 2 - dim_];
            }
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    template<typename Num2, unsigned Len2, unsigned Dim2>
    ArrayND<Numeric,Len,Dim> ArrayND<Numeric,Len,Dim>::dot(
        const ArrayND<Num2,Len2,Dim2>& r) const
    {
        if (!dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::dot method "
            "can not be used with array of 0 rank");
        if (!r.dim_) throw npstat::NpstatInvalidArgument(
            "npstat::ArrayND::dot method "
            "can not be used with argument array of 0 rank");
        if (shape_[dim_ - 1] != r.shape_[0]) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::dot: incompatible argument array shape");

        if (dim_ == 1 && r.dim_ == 1)
        {
            // Special case: the result is of 0 rank
            ArrayND<Numeric,Len,Dim> result(static_cast<unsigned*>(0), 0U);
            Numeric sum = Numeric();
            const unsigned imax = shape_[0];
            for (unsigned i=0; i<imax; ++i)
                sum += data_[i]*r.data_[i];
            result() = sum;
            return result;
        }
        else
        {
            unsigned newshapeBuf[2*Dim];
            unsigned *newshape = makeBuffer(dim_+r.dim_-2, newshapeBuf, 2*Dim);
            copyBuffer(newshape, shape_, dim_-1);
            copyBuffer(newshape+(dim_-1), r.shape_+1, r.dim_-1);
            ArrayND<Numeric,Len,Dim> result(newshape, dim_+r.dim_-2);

            dotProductLoop(0U, 0UL, 0UL, 0UL, r, result);

            destroyBuffer(newshape, newshapeBuf);
            return result;            
        }
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline unsigned ArrayND<Numeric,Len,Dim>::span(const unsigned dim) const
    {
        if (dim >= dim_)
            throw npstat::NpstatOutOfRange(
                "In npstat::ArrayND::span: dimension number is out of range");
        return shape_[dim];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    unsigned ArrayND<Numeric,Len,Dim>::maximumSpan() const
    {
        unsigned maxspan = 0;
        for (unsigned i=0; i<dim_; ++i)
            if (shape_[i] > maxspan)
                maxspan = shape_[i];
        return maxspan;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    unsigned ArrayND<Numeric,Len,Dim>::minimumSpan() const
    {
        if (dim_ == 0)
            return 0U;
        unsigned minspan = shape_[0];
        for (unsigned i=1; i<dim_; ++i)
            if (shape_[i] < minspan)
                minspan = shape_[i];
        return minspan;
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl() const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"cl\"");
        if (dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 0 array)");
        return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0) const
    {
        if (1U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 1 array)");
        return data_[coordToIndex(i0, 0)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1) const
    {
        if (2U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 2 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2) const 
    {
        if (3U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 3 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3) const 
    {
        if (4U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 4 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4) const 
    {
        if (5U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 5 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5) const 
    {
        if (6U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 6 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6) const 
    {
        if (7U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 7 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6,
        const double i7) const 
    {
        if (8U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 8 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)*strides_[6] + 
                     coordToIndex(i7, 7)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6,
        const double i7,
        const double i8) const 
    {
        if (9U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 9 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)*strides_[6] + 
                     coordToIndex(i7, 7)*strides_[7] + 
                     coordToIndex(i8, 8)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline const Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6,
        const double i7,
        const double i8,
        const double i9) const 
    {
        if (10U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 10 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)*strides_[6] + 
                     coordToIndex(i7, 7)*strides_[7] + 
                     coordToIndex(i8, 8)*strides_[8] +
                     coordToIndex(i9, 9)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl()
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"cl\"");
        if (dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 0 array)");
        return localData_[0];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0)
    {
        if (1U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 1 array)");
        return data_[coordToIndex(i0, 0)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1)
    {
        if (2U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 2 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2) 
    {
        if (3U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 3 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3) 
    {
        if (4U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 4 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4) 
    {
        if (5U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 5 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5) 
    {
        if (6U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 6 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6) 
    {
        if (7U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 7 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6,
        const double i7) 
    {
        if (8U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 8 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)*strides_[6] + 
                     coordToIndex(i7, 7)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6,
        const double i7,
        const double i8) 
    {
        if (9U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 9 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)*strides_[6] + 
                     coordToIndex(i7, 7)*strides_[7] + 
                     coordToIndex(i8, 8)];
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    inline Numeric& ArrayND<Numeric,Len,Dim>::cl(
        const double i0,
        const double i1,
        const double i2,
        const double i3,
        const double i4,
        const double i5,
        const double i6,
        const double i7,
        const double i8,
        const double i9) 
    {
        if (10U != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::cl: wrong # of args (not rank 10 array)");
        return data_[coordToIndex(i0, 0)*strides_[0] + 
                     coordToIndex(i1, 1)*strides_[1] + 
                     coordToIndex(i2, 2)*strides_[2] + 
                     coordToIndex(i3, 3)*strides_[3] + 
                     coordToIndex(i4, 4)*strides_[4] + 
                     coordToIndex(i5, 5)*strides_[5] +
                     coordToIndex(i6, 6)*strides_[6] + 
                     coordToIndex(i7, 7)*strides_[7] + 
                     coordToIndex(i8, 8)*strides_[8] +
                     coordToIndex(i9, 9)];
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    const char* ArrayND<Numeric,StackLen,StackDim>::classname()
    {
        static const std::string name(
            gs::template_class_name<Numeric>("npstat::ArrayND"));
        return name.c_str();
    }

    template<typename Numeric, unsigned StackLen, unsigned StackDim>
    bool ArrayND<Numeric,StackLen,StackDim>::write(std::ostream& os) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"write\"");
        gs::write_pod_vector(os, shape());
        return !os.fail() && 
            (dim_ ? gs::write_array(os, data_, len_) : 
             gs::write_item(os, localData_[0], false));
    }

    template<typename Numeric, unsigned Len, unsigned Dim>
    void ArrayND<Numeric,Len,Dim>::restore(
        const gs::ClassId& id, std::istream& in, ArrayND* array)
    {
        static const gs::ClassId current(gs::ClassId::makeId<ArrayND<Numeric,Len,Dim> >());
        current.ensureSameId(id);

        ArrayShape rshape;
        gs::read_pod_vector(in, &rshape);
        if (in.fail()) throw gs::IOReadFailure(
            "In npstat::ArrayND::restore: input stream failure (checkpoint 0)");

        assert(array);
        array->uninitialize();
        array->dim_ = rshape.size();
        array->shapeIsKnown_ = true;
        array->len_ = 1UL;
        if (array->dim_)
        {
            array->shape_ = makeBuffer(array->dim_, array->localShape_, Dim);
            for (unsigned i=0; i<array->dim_; ++i)
            {
                array->shape_[i] = rshape[i];
                assert(array->shape_[i]);
                array->len_ *= array->shape_[i];
            }
            array->buildStrides();
            array->data_ = makeBuffer(array->len_, array->localData_, Len);
            gs::read_array(in, array->data_, array->len_);
        }
        else
            gs::restore_item(in, array->localData_, false);
        if (in.fail()) throw gs::IOReadFailure(
            "In npstat::ArrayND::restore: input stream failure (checkpoint 1)");
    }

    template<typename Numeric, unsigned Len, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,Len,StackDim>::exportSubrange(
        const unsigned* corner, const unsigned lenCorner,
        ArrayND<Num2, Len2, Dim2>* out) const
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"exportSubrange\"");
        if (dim_ != lenCorner) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::exportSubrange: incompatible corner index length");
        assert(out);
        if (!out->shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::exportSubrange: uninitialized argument array");
        if (out->dim_ != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::exportSubrange: incompatible argument array rank");

        if (dim_)
        {
            assert(corner);
            if (dim_ > CHAR_BIT*sizeof(unsigned long))
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::exportSubrange: "
                    "array rank is too large");
            unsigned toBuf[CHAR_BIT*sizeof(unsigned long)];
            clearBuffer(toBuf, dim_);
            (const_cast<ArrayND*>(this))->commonSubrangeLoop(
                0U, 0UL, 0UL, corner, out->shape_, toBuf, *out,
                scast_assign_right<Numeric,Num2>());
        }
        else
            out->localData_[0] = static_cast<Num2>(localData_[0]);
    }

    template<typename Numeric, unsigned Len, unsigned StackDim>
    template <typename Num2, unsigned Len2, unsigned Dim2>
    void ArrayND<Numeric,Len,StackDim>::importSubrange(
        const unsigned* corner, const unsigned lenCorner,
        const ArrayND<Num2, Len2, Dim2>& from)
    {
        if (!shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "Initialize npstat::ArrayND before calling method \"importSubrange\"");
        if (dim_ != lenCorner) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::importSubrange: incompatible corner index length");
        if (!from.shapeIsKnown_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::importSubrange: uninitialized argument array");
        if (from.dim_ != dim_) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayND::importSubrange: incompatible argument array rank");

        if (dim_)
        {
            assert(corner);
            if (dim_ > CHAR_BIT*sizeof(unsigned long))
                throw npstat::NpstatInvalidArgument(
                    "In npstat::ArrayND::importSubrange: "
                    "array rank is too large");
            unsigned toBuf[CHAR_BIT*sizeof(unsigned long)];
            clearBuffer(toBuf, dim_);
            commonSubrangeLoop(0U, 0UL, 0UL, corner, from.shape_, toBuf,
                               const_cast<ArrayND<Num2, Len2, Dim2>&>(from),
                               scast_assign_left<Numeric,Num2>());
        }
        else
            localData_[0] = static_cast<Numeric>(from.localData_[0]);
    }
}


#endif // NPSTAT_ARRAYND_HH_

