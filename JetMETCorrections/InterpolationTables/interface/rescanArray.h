#ifndef NPSTAT_RESCANARRAY_HH_
#define NPSTAT_RESCANARRAY_HH_

/*!
// \file rescanArray.h
//
// \brief Fill a multidimensional array using values from another array
//        with a different shape
//
// Author: I. Volobouev
//
// October 2010
*/

#include "JetMETCorrections/InterpolationTables/interface/ArrayND.h"

namespace npstat {
    /**
    // A utility for filling one array using values of another. The
    // array shapes do not have to be the same but the ranks have to be.
    // Roughly, the arrays are treated as values of histogram bins inside
    // the unit box. The array "to" is filled either with the closest bin
    // value of the array "from" or with an interpolated value (if 
    // "interpolationDegree" parameter is not 0).
    //
    // interpolationDegree parameter must be one of 0, 1, or 3.
    */
    template<typename Num1, unsigned Len1, unsigned Dim1,
             typename Num2, unsigned Len2, unsigned Dim2>
    void rescanArray(const ArrayND<Num1,Len1,Dim1>& from,
                     ArrayND<Num2,Len2,Dim2>* to,
                     unsigned interpolationDegree=0);
}

#include <cassert>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "JetMETCorrections/InterpolationTables/interface/LinearMapper1d.h"

namespace npstat {
    namespace Private {
        template<typename Num1, unsigned Len1, unsigned Dim1,
                 typename Num2, unsigned Len2, unsigned Dim2>
        class ArrayMapper
        {
        public:
            ArrayMapper(const ArrayND<Num1,Len1,Dim1>& from,
                        const ArrayND<Num2,Len2,Dim2>& to,
                        const unsigned interpolationDegree)
                : mapped_(from.rank()),
                  from_(from),
                  dim_(from.rank()),
                  ideg_(interpolationDegree)
            {
                assert(dim_ == to.rank());
                if (dim_)
                {
                    mappers_.reserve(dim_);
                    for (unsigned i=0; i<dim_; ++i)
                        mappers_.push_back(LinearMapper1d(-0.5, -0.5,
                                                          to.span(i) - 0.5,
                                                          from.span(i) - 0.5));
                }
            }

            Num1 operator()(const unsigned* index, unsigned indexLen) const
            {
                if (dim_)
                {
                    const LinearMapper1d* m = &mappers_[0];
                    double* coords = &mapped_[0];
                    for (unsigned i=0; i<dim_; ++i)
                        coords[i] = m[i](index[i]);
                    switch (ideg_)
                    {
                    case 0U:
                        return from_.closest(coords, dim_);

                    case 1U:
                        return from_.interpolate1(coords, dim_);

                    case 3U:
                        return from_.interpolate3(coords, dim_);

                    default:
                        assert(0);
                    }
                }
                return from_();
            }

        private:
            std::vector<LinearMapper1d> mappers_;
            mutable std::vector<double> mapped_;
            const ArrayND<Num1,Len1,Dim1>& from_;
            unsigned dim_;
            unsigned ideg_;
        };
    }

    template<typename Num1, unsigned Len1, unsigned Dim1,
             typename Num2, unsigned Len2, unsigned Dim2>
    void rescanArray(const ArrayND<Num1,Len1,Dim1>& from,
                     ArrayND<Num2,Len2,Dim2>* to,
                     const unsigned interpolationDegree)
    {
        assert(to);
        if (from.rank() != to->rank()) throw npstat::NpstatInvalidArgument(
            "In npstat::rescanArray: incompatible dimensionalities "
            "of input and output arrays");
        if (!(interpolationDegree == 0U ||
              interpolationDegree == 1U ||
              interpolationDegree == 3U)) throw npstat::NpstatInvalidArgument(
                  "In npstat::rescanArray: unsupported interpolation degree");
        to->functorFill(Private::ArrayMapper<Num1,Len1,Dim1,Num2,Len2,Dim2>(
                            from, *to, interpolationDegree));
    }
}


#endif // NPSTAT_RESCANARRAY_HH_

