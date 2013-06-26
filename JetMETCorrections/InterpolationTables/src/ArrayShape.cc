#include <cassert>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "JetMETCorrections/InterpolationTables/interface/ArrayShape.h"

namespace npstat {
    ArrayShape makeShape()
    {
        return ArrayShape();
    }

    ArrayShape makeShape(unsigned i0)
    {
        return ArrayShape(1, i0);
    }

    ArrayShape makeShape(unsigned i0, unsigned i1)
    {
        ArrayShape s;
        s.reserve(2);
        s.push_back(i0);
        s.push_back(i1);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2)
    {
        ArrayShape s;
        s.reserve(3);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3)
    {
        ArrayShape s;
        s.reserve(4);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        s.push_back(i3);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4)
    {
        ArrayShape s;
        s.reserve(5);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        s.push_back(i3);
        s.push_back(i4);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5)
    {
        ArrayShape s;
        s.reserve(6);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        s.push_back(i3);
        s.push_back(i4);
        s.push_back(i5);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6)
    {
        ArrayShape s;
        s.reserve(7);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        s.push_back(i3);
        s.push_back(i4);
        s.push_back(i5);
        s.push_back(i6);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6, unsigned i7)
    {
        ArrayShape s;
        s.reserve(8);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        s.push_back(i3);
        s.push_back(i4);
        s.push_back(i5);
        s.push_back(i6);
        s.push_back(i7);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                         unsigned i8)
    {
        ArrayShape s;
        s.reserve(9);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        s.push_back(i3);
        s.push_back(i4);
        s.push_back(i5);
        s.push_back(i6);
        s.push_back(i7);
        s.push_back(i8);
        return s;
    }

    ArrayShape makeShape(unsigned i0, unsigned i1, unsigned i2, unsigned i3,
                         unsigned i4, unsigned i5, unsigned i6, unsigned i7,
                         unsigned i8, unsigned i9)
    {
        ArrayShape s;
        s.reserve(10);
        s.push_back(i0);
        s.push_back(i1);
        s.push_back(i2);
        s.push_back(i3);
        s.push_back(i4);
        s.push_back(i5);
        s.push_back(i6);
        s.push_back(i7);
        s.push_back(i8);
        s.push_back(i9);
        return s;
    }

    ArrayShape makeShape(const unsigned* indices, const unsigned nIndices)
    {
        ArrayShape s;
        if (nIndices)
        {
            assert(indices);
            s.reserve(nIndices);
            for (unsigned i=0; i<nIndices; ++i)
                s.push_back(indices[i]);
        }
        return s;
    }

    ArrayShape doubleShape(const ArrayShape& inputShape)
    {
        ArrayShape s(inputShape);
        const unsigned len = s.size();
        for (unsigned i=0; i<len; ++i)
            s[i] *= 2U;
        return s;
    }

    ArrayShape halfShape(const ArrayShape& inputShape)
    {
        ArrayShape s(inputShape);
        const unsigned len = s.size();
        for (unsigned i=0; i<len; ++i)
        {
            if (!(s[i] % 2U == 0)) throw npstat::NpstatInvalidArgument(
                "In npstat::halfShape: array span must be "
                "even in each dimension");
            s[i] /= 2U;
        }
        return s;
    }

    bool isSubShape(const ArrayShape& sh1, const ArrayShape& sh2)
    {
        const unsigned len = sh1.size();
        if (len != sh2.size())
            return false;
        for (unsigned i=0; i<len; ++i)
            if (sh1[i] > sh2[i])
                return false;
        return true;
    }
}
