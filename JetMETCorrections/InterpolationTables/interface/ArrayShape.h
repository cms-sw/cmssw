#ifndef NPSTAT_ARRAYSHAPE_HH_
#define NPSTAT_ARRAYSHAPE_HH_

/*!
// \file ArrayShape.h
//
// \brief Utilities for defining shapes of multidimensional arrays
//
// Author: I. Volobouev
//
// October 2009
*/

#include <vector>

namespace npstat {
    /**
    // This type will be used to specify
    // array length in each dimension
    */
    typedef std::vector<unsigned> ArrayShape;

    //@{
    /**
    // This convenience function will construct
    // an array shape using an explicit list of indices
    */
    ArrayShape makeShape();
    ArrayShape makeShape(unsigned i0);
    ArrayShape makeShape(unsigned i0, unsigned i1);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6, unsigned i7);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                         unsigned i8);
    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                         unsigned i8, unsigned i9);
    ArrayShape makeShape(const unsigned* indices, unsigned nIndices);
    //@}

    /** Multiply the sise in each dimension by 2 */
    ArrayShape doubleShape(const ArrayShape& inputShape);

    /** Divide the size in each dimension by 2 (generate dynamic fault if odd) */
    ArrayShape halfShape(const ArrayShape& inputShape);

    /**
    // This function returns true if the number of elements is
    // the same in both vectors and every element of the first vector
    // does not exceed corresponding element of the second
    */
    bool isSubShape(const ArrayShape& sh1, const ArrayShape& sh2);
}

#endif // NPSTAT_ARRAYSHAPE_HH_

