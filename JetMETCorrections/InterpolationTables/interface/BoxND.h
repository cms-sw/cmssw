#ifndef NPSTAT_BOXND_HH_
#define NPSTAT_BOXND_HH_

/*!
// \file BoxND.h
//
// \brief Template to represent rectangles, boxes, and hyperboxes
//
// Author: I. Volobouev
//
// March 2010
*/

#include <vector>

#include "Alignment/Geners/interface/ClassId.hh"
#include "JetMETCorrections/InterpolationTables/interface/Interval.h"

namespace npstat {
    /**
    // Class to represent rectangles, boxes, and hyperboxes
    */
    template <typename Numeric>
    struct BoxND : public std::vector<Interval<Numeric> >
    {
        /** Default constructor makes a 0-dimensional box */
        inline BoxND() {}

        /** Interval in each dimension is made by its default constructor */
        inline explicit BoxND(const unsigned long dim) :
            std::vector<Interval<Numeric> >(dim) {}

        /** Use the same interval in each dimension */
        inline BoxND(const unsigned long dim, const Interval<Numeric>& v) :
            std::vector<Interval<Numeric> >(dim, v) {}

        /**
        // Constructor where one of the limits will be 0 and the other
        // will be generated from the given vector (which also determines
        // the dimensionality)
        */
        template <typename Num2>
        explicit BoxND(const std::vector<Num2>& limits);

        /** Converting constructor */
        template <typename Num2>
        explicit BoxND(const BoxND<Num2>& r);

        /**
        // Get the data from a box of a different type. This method
        // works essentially as a converting assignment operator.
        */
        template <typename Num2>
        BoxND& copyFrom(const BoxND<Num2>& r);

        /** Box dimensionality */
        inline unsigned long dim() const {return this->size();}

        /** Box volume */
        Numeric volume() const;

        /**
        // Midpoint for every coordinate. The size of the "coord"
        // array should be at least as large as the box dimensionality.
        */
        void getMidpoint(Numeric* coord, unsigned long coordLen) const;

        //@{
        /**
        // This method return "true" if the corresponding function
        // of the Interval returns "true" for every coordinate.
        // There must be an automatic conversion from Num2 type into Numeric.
        */
        template <typename Num2>
        bool isInsideLower(const Num2* coord, unsigned long coordLen) const;
        template <typename Num2>
        bool isInsideUpper(const Num2* coord, unsigned long coordLen) const;
        template <typename Num2>
        bool isInsideWithBounds(const Num2* coord, unsigned long coordLen) const;
        template <typename Num2>
        bool isInside(const Num2* coord, unsigned long coordLen) const;
        //@}

        //@{
        /** Scaling of all limits by a constant */
        BoxND& operator*=(double r);
        BoxND& operator/=(double r);
        //@}

        //@{
        /** Scaling by a different constant in each dimension */
        BoxND& operator*=(const std::vector<double>& scales);
        BoxND& operator/=(const std::vector<double>& scales);
        //@}

        /**
        // Scaling of all limits by a constant in such a way that the midpoint
        // remains unchanged
        */
        BoxND& expand(double r);

        //@{
        /**
        // Scaling of all limits in such a way that the midpoint
        // remains unchanged, using a different scaling factor
        // in each dimension
        */
        BoxND& expand(const std::vector<double>& scales);
        BoxND& expand(const double* scales, unsigned long lenScales);
        //@}

        //@{
        /** Shifting this object */
        template <typename Num2>
        BoxND& operator+=(const std::vector<Num2>& shifts);
        template <typename Num2>
        BoxND& operator-=(const std::vector<Num2>& shifts);
        template <typename Num2>
        BoxND& shift(const Num2* shifts, unsigned long lenShifts);
        //@}

        /** Moving this object so that the midpoint is (0, 0, ..., 0) */
        BoxND& moveToOrigin();

        /** Overlap volume with another box */
        Numeric overlapVolume(const BoxND& r) const;

        /** A faster way to calculate overlapVolume(r)/volume() */
        double overlapFraction(const BoxND& r) const;

        /** Box with lower limit 0 and upper limit 1 in all coordinates */
        static BoxND unitBox(unsigned long ndim);

        /**
        // Box with lower limit -1 and upper limit 1 in all coordinates.
        // Note that this will produce nonsense in case the Numeric type
        // is unsigned.
        */
        static BoxND sizeTwoBox(unsigned long ndim);

        /**
        // Box with all upper limits set to maximum possible Numeric
        // number and with lower limits set to negative maximum (this
        // will not work with unsigned long types)
        */
        static BoxND allSpace(unsigned long ndim);

        //@{
        /** Methods related to I/O */
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static const char* classname();
        static inline unsigned version() {return 1;}
        static void restore(const gs::ClassId& id, std::istream& in, BoxND* box);
    };
}

//@{
/** Binary comparison for equality */
template <typename Numeric>
bool operator==(const npstat::BoxND<Numeric>& l, const npstat::BoxND<Numeric>& r);

template <typename Numeric>
bool operator!=(const npstat::BoxND<Numeric>& l, const npstat::BoxND<Numeric>& r);
//@}

#include <limits>
#include <cassert>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "Alignment/Geners/interface/GenericIO.hh"

namespace npstat {
    template <typename Numeric>
    template <typename Num2>
    BoxND<Numeric>::BoxND(const BoxND<Num2>& r)
    {
        const unsigned long dim = r.size();
        if (dim)
        {
            this->reserve(dim);
            for (unsigned long i=0; i<dim; ++i)
            {
                const Interval<Num2>& ri(r[i]);
                this->push_back(Interval<Numeric>(ri.min(), ri.max()));
            }
        }
    }

    template <typename Numeric>
    template <typename Num2>
    BoxND<Numeric>::BoxND(const std::vector<Num2>& limits)
    {
        const unsigned long dim = limits.size();
        if (dim)
        {
            this->reserve(dim);
            Numeric zero = Numeric();
            for (unsigned long i=0; i<dim; ++i)
            {
                const Numeric value(static_cast<Numeric>(limits[i]));
                if (value >= zero)
                    this->push_back(Interval<Numeric>(zero, value));
                else
                    this->push_back(Interval<Numeric>(value, zero));
            }
        }
    }

    template <typename Numeric>
    template <typename Num2>
    BoxND<Numeric>& BoxND<Numeric>::copyFrom(const BoxND<Num2>& r)
    {
        if ((void *)this == (void *)(&r))
            return *this;
        const unsigned long n = r.size();
        this->clear();
        this->reserve(n);
        for (unsigned long i=0; i<n; ++i)
        {
            const Interval<Num2>& ir(r[i]);
            this->push_back(Interval<Numeric>(ir.min(), ir.max()));
        }
        return *this;
    }

    template <typename Numeric>
    Numeric BoxND<Numeric>::volume() const
    {
        Numeric v(static_cast<Numeric>(1));
        const unsigned long mydim = this->size();
        for (unsigned long i=0U; i<mydim; ++i)
            v *= (*this)[i].length();
        return v;
    }

    template <typename Numeric>
    Numeric BoxND<Numeric>::overlapVolume(const BoxND& r) const
    {
        const unsigned long mydim = this->size();
        if (mydim == r.size())
        {
            Numeric v(static_cast<Numeric>(1));
            for (unsigned long i=0U; i<mydim; ++i)
                v *= (*this)[i].overlapLength(r[i]);
            return v;
        }
        else
            return static_cast<Numeric>(0);
    }

    template <typename Numeric>
    double BoxND<Numeric>::overlapFraction(const BoxND& r) const
    {
        const unsigned long mydim = this->size();
        if (mydim == r.size())
        {
            double f = 1.0;
            for (unsigned long i=0U; i<mydim; ++i)
                f *= (*this)[i].overlapFraction(r[i]);
            return f;
        }
        else
            return 0.0;
    }

    template <typename Numeric>
    void BoxND<Numeric>::getMidpoint(Numeric* coord,
                                     const unsigned long coordLen) const
    {
        const unsigned long mydim = this->size();
        if (coordLen < mydim) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::getMidpoint: insufficient output buffer length");
        if (mydim)
        {
            assert(coord);
            for (unsigned  long i=0U; i<mydim; ++i)
                coord[i] = (*this)[i].midpoint();
        }
    }

    template <typename Numeric>
    template <typename Num2>
    bool BoxND<Numeric>::isInsideLower(const Num2* coords,
                                       const unsigned long coordLen) const
    {
        if (coordLen != this->size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::isInsideLower: "
            "incompatible point dimensionality");
        const Interval<Numeric>* myptr = &(*this)[0];
        for (unsigned long i=0; i<coordLen; ++i)
            if (!myptr[i].isInsideLower(coords[i]))
                return false;
        return true;
    }

    template <typename Numeric>
    template <typename Num2>
    bool BoxND<Numeric>::isInsideUpper(const Num2* coords,
                                       const unsigned long coordLen) const
    {
        if (coordLen != this->size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::isInsideUpper: "
            "incompatible point dimensionality");
        const Interval<Numeric>* myptr = &(*this)[0];
        for (unsigned long i=0; i<coordLen; ++i)
            if (!myptr[i].isInsideUpper(coords[i]))
                return false;
        return true;
    }

    template <typename Numeric>
    template <typename Num2>
    bool BoxND<Numeric>::isInsideWithBounds(const Num2* coords,
                                            const unsigned long coordLen) const
    {
        if (coordLen != this->size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::isInsideWithBounds: "
            "incompatible point dimensionality");
        const Interval<Numeric>* myptr = &(*this)[0];
        for (unsigned long i=0; i<coordLen; ++i)
            if (!myptr[i].isInsideWithBounds(coords[i]))
                return false;
        return true;
    }

    template <typename Numeric>
    template <typename Num2>
    bool BoxND<Numeric>::isInside(const Num2* coords,
                                  const unsigned long coordLen) const
    {
        if (coordLen != this->size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::isInside: incompatible point dimensionality");
        const Interval<Numeric>* myptr = &(*this)[0];
        for (unsigned long i=0; i<coordLen; ++i)
            if (!myptr[i].isInside(coords[i]))
                return false;
        return true;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::operator*=(const double r)
    {
        const unsigned long mydim = this->size();
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i] *= r;
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::moveToOrigin()
    {
        const unsigned long mydim = this->size();
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i].moveMidpointTo0();
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::expand(const double r)
    {
        const unsigned long mydim = this->size();
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i].expand(r);
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::operator*=(
        const std::vector<double>& scales)
    {
        const unsigned long mydim = this->size();
        if (mydim != scales.size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::operator*=: "
            "incompatible argument dimensionality");
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i] *= scales[i];
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::expand(
        const std::vector<double>& scales)
    {
        const unsigned long mydim = this->size();
        if (mydim != scales.size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::expand: incompatible argument dimensionality");
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i].expand(scales[i]);
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::expand(
        const double* scales, const unsigned long lenScales)
    {
        const unsigned long mydim = this->size();
        if (mydim != lenScales) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::expand: incompatible argument dimensionality");
        if (mydim)
        {
            assert(scales);
            for (unsigned long i=0; i<mydim; ++i)
                (*this)[i].expand(scales[i]);
        }
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::operator/=(const double r)
    {
        const unsigned long mydim = this->size();
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i] /= r;
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric>& BoxND<Numeric>::operator/=(
        const std::vector<double>& scales)
    {
        const unsigned long mydim = this->size();
        if (mydim != scales.size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::operator/=: "
            "incompatible argument dimensionality");
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i] /= scales[i];
        return *this;
    }

    template <typename Numeric>
    template <typename Num2>
    BoxND<Numeric>& BoxND<Numeric>::operator+=(const std::vector<Num2>& shifts)
    {
        const unsigned long mydim = this->size();
        if (mydim != shifts.size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::operator+=: "
            "incompatible argument dimensionality");
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i] += static_cast<Numeric>(shifts[i]);
        return *this;
    }

    template <typename Numeric>
    template <typename Num2>
    BoxND<Numeric>& BoxND<Numeric>::shift(
        const Num2* shifts, const unsigned long shiftsLen)
    {
        const unsigned long mydim = this->size();
        if (mydim != shiftsLen) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::shift: incompatible argument dimensionality");
        if (mydim)
        {
            assert(shifts);
            for (unsigned long i=0; i<mydim; ++i)
                (*this)[i] += static_cast<Numeric>(shifts[i]);
        }
        return *this;
    }

    template <typename Numeric>
    template <typename Num2>
    BoxND<Numeric>& BoxND<Numeric>::operator-=(const std::vector<Num2>& shifts)
    {
        const unsigned long mydim = this->size();
        if (mydim != shifts.size()) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxND::operator-=: "
            "incompatible argument dimensionality");
        for (unsigned long i=0; i<mydim; ++i)
            (*this)[i] -= static_cast<Numeric>(shifts[i]);
        return *this;
    }

    template <typename Numeric>
    BoxND<Numeric> BoxND<Numeric>::unitBox(const unsigned long ndim)
    {
        Interval<Numeric> unit(static_cast<Numeric>(0),
                               static_cast<Numeric>(1));
        return BoxND<Numeric>(ndim, unit);
    }

    template <typename Numeric>
    BoxND<Numeric> BoxND<Numeric>::sizeTwoBox(const unsigned long ndim)
    {
        const Numeric one = static_cast<Numeric>(1);
        Interval<Numeric> i(-one, one);
        return BoxND<Numeric>(ndim, i);
    }

    template <typename Numeric>
    BoxND<Numeric> BoxND<Numeric>::allSpace(const unsigned long ndim)
    {
        const Numeric maxval = std::numeric_limits<Numeric>::max();
        Interval<Numeric> i(-maxval, maxval);
        return BoxND<Numeric>(ndim, i);
    }

    template<typename Numeric>
    const char* BoxND<Numeric>::classname()
    {
        static const std::string na(gs::template_class_name<Numeric>("npstat::BoxND"));
        return na.c_str();
    }

    template<typename Numeric>
    bool BoxND<Numeric>::write(std::ostream& of) const
    {
        const unsigned long mydim = this->size();
        std::vector<Numeric> limits;
        limits.reserve(2UL*mydim);
        for (unsigned long i=0; i<mydim; ++i)
        {
            limits.push_back((*this)[i].min());
            limits.push_back((*this)[i].max());
        }
        return gs::write_item(of, limits);
    }

    template<typename Numeric>
    void BoxND<Numeric>::restore(const gs::ClassId& id, std::istream& in, BoxND* b)
    {
        static const gs::ClassId current(gs::ClassId::makeId<BoxND<Numeric> >());
        current.ensureSameId(id);

        std::vector<Numeric> limits;
        gs::restore_item(in, &limits);
        if (in.fail())
            throw gs::IOReadFailure("In npstat::BoxND::restore: input stream failure");
        const unsigned long nlimits = limits.size();
        if (nlimits % 2UL)
            throw gs::IOInvalidData("In npstat::BoxND::restore: bad limits");
        assert(b);
        b->clear();
        b->reserve(nlimits/2UL);
        for (unsigned long i=0; i<nlimits/2UL; ++i)
            b->push_back(npstat::Interval<Numeric>(limits[2U*i], limits[2U*i+1U]));
    }
}

template <typename Numeric>
bool operator==(const npstat::BoxND<Numeric>& l, const npstat::BoxND<Numeric>& r)
{
    const unsigned long dim = l.size();
    if (dim != r.size())
        return false;
    for (unsigned long i=0; i<dim; ++i)
        if (l[i] != r[i])
            return false;
    return true;
}

template <typename Numeric>
bool operator!=(const npstat::BoxND<Numeric>& l, const npstat::BoxND<Numeric>& r)
{
    return !(l == r);
}


#endif // NPSTAT_BOXND_HH_

