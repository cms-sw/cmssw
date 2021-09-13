#ifndef NPSTAT_BOXNDSCANNER_HH_
#define NPSTAT_BOXNDSCANNER_HH_

/*!
// \file BoxNDScanner.h
//
// \brief Iteration over uniformly spaced coordinates inside
//        a multidimensional box
//
// Author: I. Volobouev
//
// March 2010
*/

#include "JetMETCorrections/InterpolationTables/interface/BoxND.h"

namespace npstat {
   /**
    * A class for iterating over all coordinates in a multidimensional box
    * (but not a full-fledeged iterator). The expected usage pattern is as
    * follows:
    *
    * @code
    *   double* coords = ...   (the buffer size should be at least box.dim())
    *   for (BoxNDScanner<double> scan(box,shape); scan.isValid(); ++scan)
    *   {
    *       scan.getCoords(coords, coordsBufferSize);
    *       .... Do what is necessary with coordinates ....
    *       .... Extract linear bin number:  ..............
    *       scan.state();
    *   }
    * @endcode
    *
    * The coordinates will be in the middle of the bins (imagine
    * a multivariate histogram with boundaries defined by the given box).
    */
    template <typename Numeric>
    class BoxNDScanner
    {
    public:
        //@{
        /** 
        // Constructor from a bounding box and a multidimensional
        // array shape
        */
        inline BoxNDScanner(const BoxND<Numeric>& box,
                            const std::vector<unsigned>& shape)
            : box_(box), state_(0UL) 
            {initialize(shape.empty() ? static_cast<unsigned*>(0) : 
                        &shape[0], shape.size());}

        inline BoxNDScanner(const BoxND<Numeric>& box,
                            const unsigned* shape, const unsigned lenShape)
            : box_(box), state_(0UL) {initialize(shape, lenShape);}
        //@}

        /** Dimensionality of the scan */
        inline unsigned dim() const {return box_.dim();}

        /** Retrieve current state (i.e., linear index of the scan) */
        inline unsigned long state() const {return state_;}

        /** Maximum possible state (i.e., linear index of the scan) */
        inline unsigned long maxState() const {return maxState_;}

        /** Returns false when iteration is complete */
        inline bool isValid() const {return state_ < maxState_;}

        /** Retrieve current coordinates inside the box */
        void getCoords(Numeric* x, unsigned nx) const;

        /** Retrieve current multidimensional index */
        void getIndex(unsigned* index, unsigned indexBufferLen) const;

        /** Reset the state (as if the object has just been constructed) */
        inline void reset() {state_ = 0UL;}

        /** Prefix increment */
        inline BoxNDScanner& operator++()
            {if (state_ < maxState_) ++state_; return *this;}

        /** Postfix increment (distinguished by the dummy "int" parameter) */
        inline void operator++(int) {if (state_ < maxState_) ++state_;}

        /** Set the state directly */
        inline void setState(const unsigned long state)
            {state_ = state <= maxState_ ? state : maxState_;}

    private:
        BoxNDScanner();

        void initialize(const unsigned* shape, unsigned lenShape);

        BoxND<Numeric> box_;
        std::vector<unsigned long> strides_;
        std::vector<double> bw_;
        unsigned long state_;
        unsigned long maxState_;
    };
}

#include <cassert>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

namespace npstat {
    template <typename Numeric>
    void BoxNDScanner<Numeric>::initialize(const unsigned* shape,
                                           const unsigned dim)
    {
        if (!(dim && box_.dim() == dim)) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxNDScanner::initialize: incompatible scan shape");
        assert(shape);
        for (unsigned j=0; j<dim; ++j)
            if (!shape[j]) throw npstat::NpstatInvalidArgument(
                "In npstat::BoxNDScanner::initialize: "
                "number of scans must be positive in each dimension");

        strides_.resize(dim);
        strides_[dim - 1] = 1UL;
        for (unsigned j=dim - 1; j>0; --j)
            strides_[j - 1] = strides_[j]*shape[j];
        maxState_ = strides_[0]*shape[0];

        bw_.reserve(dim);
        for (unsigned j=0; j<dim; ++j)
            bw_.push_back(box_[j].length()*1.0/shape[j]);
    }

    template <typename Numeric>
    void BoxNDScanner<Numeric>::getCoords(Numeric* x, const unsigned nx) const
    {
        const unsigned dim = strides_.size();
        if (nx < dim) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxNDScanner::getCoords: "
            "insufficient length of the output buffer");
        if (state_ >= maxState_) throw npstat::NpstatRuntimeError(
            "In npstat::BoxNDScanner::getCoords: invalid scanner state");
        assert(x);

        unsigned long l = state_;
        for (unsigned i=0; i<dim; ++i)
        {
            unsigned long idx = l / strides_[i];
            x[i] = box_[i].min() + (idx + 0.5)*bw_[i];
            l -= (idx * strides_[i]);
        }
    }

    template <typename Numeric>
    void BoxNDScanner<Numeric>::getIndex(unsigned* ix, const unsigned nx) const
    {
        const unsigned dim = strides_.size();
        if (nx < dim) throw npstat::NpstatInvalidArgument(
            "In npstat::BoxNDScanner::getIndex: "
            "insufficient length of the output buffer");
        if (state_ >= maxState_) throw npstat::NpstatRuntimeError(
            "In npstat::BoxNDScanner::getIndex: invalid scanner state");
        assert(ix);

        unsigned long l = state_;
        for (unsigned i=0; i<dim; ++i)
        {
            unsigned long idx = l / strides_[i];
            ix[i] = static_cast<unsigned>(idx);
            l -= (idx * strides_[i]);
        }
    }
}


#endif // NPSTAT_BOXNDSCANNER_HH_

