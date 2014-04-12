#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"
#include "JetMETCorrections/InterpolationTables/interface/ArrayRange.h"

namespace npstat {
    ArrayRange::ArrayRange(const unsigned* ishape, const unsigned imax)
    {
        if (imax)
        {
            assert(ishape);
            this->reserve(imax);
            for (unsigned i=0; i<imax; ++i)
                this->push_back(Interval<unsigned>(ishape[i]));
        }
    }

    bool ArrayRange::isCompatible(const ArrayShape& ishape) const
    {
        const unsigned imax = ishape.size();
        return isCompatible(imax ? &ishape[0] : (unsigned*)0, imax);
    }

    bool ArrayRange::isCompatible(const unsigned* ishape,
                                  const unsigned imax) const
    {
        if (this->size() != imax)
            return false;
        if (imax)
        {
            assert(ishape);
            for (unsigned i=0; i<imax; ++i)
                if ((*this)[i].length() == 0U)
                    return true;
            for (unsigned i=0; i<imax; ++i)
                if ((*this)[i].max() > ishape[i])
                    return false;
        }
        return true;
    }

    bool ArrayRange::operator<(const ArrayRange& r) const
    {
        const unsigned mysize = this->size();
        const unsigned othersize = r.size();
        if (mysize < othersize)
            return true;
        if (mysize > othersize)
            return false;
        for (unsigned i=0; i<mysize; ++i)
        {
            const Interval<unsigned>& left((*this)[i]);
            const Interval<unsigned>& right(r[i]);
            if (left.min() < right.min())
                return true;
            if (left.min() > right.min())
                return false;
            if (left.max() < right.max())
                return true;
            if (left.max() > right.max())
                return false;
        }
        return false;
    }

    ArrayRange& ArrayRange::stripOuterLayer()
    {
        const unsigned mysize = this->size();
        for (unsigned i=0; i<mysize; ++i)
        {
            (*this)[i].setMin((*this)[i].min() + 1U);
            const unsigned uplim = (*this)[i].max();
            if (uplim)
                (*this)[i].setMax(uplim - 1U);
        }
        return *this;
    }

    unsigned long ArrayRange::rangeSize() const
    {
        unsigned long result = 0UL;
        const unsigned imax = this->size();
        if (imax)
        {
            result = 1UL;
            for (unsigned i=0; i<imax; ++i)
                result *= (*this)[i].length();
        }
        return result;
    }

    ArrayShape ArrayRange::shape() const
    {
        const unsigned imax = this->size();
        ArrayShape oshape(imax);
        for (unsigned i=0; i<imax; ++i)
            oshape[i] = (*this)[i].length();
        return oshape;
    }

    void ArrayRange::lowerLimits(unsigned* limits,
                                 const unsigned limitsLen) const
    {
        const unsigned imax = this->size();
        if (limitsLen < imax) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayRange::lowerLimits: "
            "insufficient size of the output buffer");
        if (imax)
        {
            assert(limits);
            const Interval<unsigned>* data = &(*this)[0];
            for (unsigned i=0; i<imax; ++i)
                limits[i] = data[i].min();
        }
    }

    void ArrayRange::upperLimits(unsigned* limits,
                                 const unsigned limitsLen) const
    {
        const unsigned imax = this->size();
        if (limitsLen < imax) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayRange::upperLimits: "
            "insufficient size of the output buffer");
        if (imax)
        {
            assert(limits);
            const Interval<unsigned>* data = &(*this)[0];
            for (unsigned i=0; i<imax; ++i)
                limits[i] = data[i].max();
        }
    }

    void ArrayRange::rangeLength(unsigned* limits,
                                 const unsigned limitsLen) const
    {
        const unsigned imax = this->size();
        if (limitsLen < imax) throw npstat::NpstatInvalidArgument(
            "In npstat::ArrayRange::rangeLength: "
            "insufficient size of the output buffer");
        if (imax)
        {
            assert(limits);
            const Interval<unsigned>* data = &(*this)[0];
            for (unsigned i=0; i<imax; ++i)
                limits[i] = data[i].length();
        }
    }
}
