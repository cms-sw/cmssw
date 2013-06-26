#ifndef NPSTAT_LININTERPOLATEDTABLEND_HH_
#define NPSTAT_LININTERPOLATEDTABLEND_HH_

/**
// \file LinInterpolatedTableND.h
//
// \brief Multilinear interpolation/extrapolation on rectangular grids
//
// Author: I. Volobouev
//
// June 2012
*/

#include <climits>
#include <vector>
#include <utility>

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"

#include "JetMETCorrections/InterpolationTables/interface/ArrayND.h"
#include "JetMETCorrections/InterpolationTables/interface/UniformAxis.h"

namespace npstat {
    /** 
    // Template for multilinear interpolation/extrapolation of values provided
    // on a rectangular coordinate grid. "Numeric" is the type stored in
    // the table. "Axis" should be one of GridAxis, UniformAxis, or DualAxis
    // classes or a user-provided class with a similar set of methods.
    */
    template <class Numeric, class Axis=UniformAxis>
    class LinInterpolatedTableND
    {
        template <typename Num2, typename Axis2>
        friend class LinInterpolatedTableND;

    public:
        typedef Numeric value_type;
        typedef Axis axis_type;

        /** 
        // Main constructor for arbitrary-dimensional interpolators.
        //
        // "axes" are the axes of the rectangular coordinate grid.
        //
        // In each pair provided by the "extrapolationType" argument,
        // the first element of the pair specifies whether the extrapolation
        // to negative infinity should be linear (if "true") or constant
        // (if "false"). The second element of the pair specifies whether
        // to extrapolate linearly to positive infinity.
        //
        // "functionLabel" is an arbitrary string which can potentially
        // be used by plotting programs and such.
        */
        LinInterpolatedTableND(
            const std::vector<Axis>& axes,
            const std::vector<std::pair<bool,bool> >& extrapolationType,
            const char* functionLabel=0);

        /** Convenience constructor for 1-d interpolators */
        LinInterpolatedTableND(const Axis& xAxis, bool leftX, bool rightX,
                               const char* functionLabel=0);

        /** Convenience constructor for 2-d interpolators */
        LinInterpolatedTableND(const Axis& xAxis, bool leftX, bool rightX,
                               const Axis& yAxis, bool leftY, bool rightY,
                               const char* functionLabel=0);

        /** Convenience constructor for 3-d interpolators */
        LinInterpolatedTableND(const Axis& xAxis, bool leftX, bool rightX,
                               const Axis& yAxis, bool leftY, bool rightY,
                               const Axis& zAxis, bool leftZ, bool rightZ,
                               const char* functionLabel=0);

        /** Convenience constructor for 4-d interpolators */
        LinInterpolatedTableND(const Axis& xAxis, bool leftX, bool rightX,
                               const Axis& yAxis, bool leftY, bool rightY,
                               const Axis& zAxis, bool leftZ, bool rightZ,
                               const Axis& tAxis, bool leftT, bool rightT,
                               const char* functionLabel=0);

        /** Convenience constructor for 5-d interpolators */
        LinInterpolatedTableND(const Axis& xAxis, bool leftX, bool rightX,
                               const Axis& yAxis, bool leftY, bool rightY,
                               const Axis& zAxis, bool leftZ, bool rightZ,
                               const Axis& tAxis, bool leftT, bool rightT,
                               const Axis& vAxis, bool leftV, bool rightV,
                               const char* functionLabel=0);

        /**
        // Converting copy constructor from interpolator
        // with another storage type
        */
        template <class Num2>
        LinInterpolatedTableND(const LinInterpolatedTableND<Num2,Axis>&);

        /**
        // Basic interpolation result. Argument point dimensionality must be
        // compatible with the interpolator dimensionality.
        */
        Numeric operator()(const double* point, unsigned dim) const;

        //@{
        /** Convenience function for low-dimensional interpolators */
        Numeric operator()(const double& x0) const;
        Numeric operator()(const double& x0, const double& x1) const;
        Numeric operator()(const double& x0, const double& x1,
                           const double& x2) const;
        Numeric operator()(const double& x0, const double& x1,
                           const double& x2, const double& x3) const;
        Numeric operator()(const double& x0, const double& x1,
                           const double& x2, const double& x3,
                           const double& x4) const;
        //@}

        //@{
        /** Examine interpolator contents */
        inline unsigned dim() const {return dim_;}
        inline const std::vector<Axis>& axes() const {return axes_;}
        inline const Axis& axis(const unsigned i) const
            {return axes_.at(i);}
        inline unsigned long length() const {return data_.length();}
        bool leftInterpolationLinear(unsigned i) const;
        bool rightInterpolationLinear(unsigned i) const;
        std::vector<std::pair<bool,bool> > interpolationType() const;
        inline const std::string& functionLabel() const
            {return functionLabel_;}
        //@}

        //@{
        /** Access the interpolator table data */
        inline const ArrayND<Numeric>& table() const {return data_;}
        inline ArrayND<Numeric>& table() {return data_;}
        //@}

        /** Convenience function for getting coordinates of the grid points */
        void getCoords(unsigned long linearIndex,
                       double* coords, unsigned coordsBufferSize) const;

        /** 
        // This method returns "true" if the method isUniform()
        // of each interpolator axis returns "true" 
        */
        bool isUniformlyBinned() const;

        /**
        // This method will return "true" if the point
        // is inside the grid limits or on the boundary
        */
        bool isWithinLimits(const double* point, unsigned dim) const;

        /** Modifier for the function label */
        inline void setFunctionLabel(const char* newlabel)
            {functionLabel_ = newlabel ? newlabel : "";}

        /**
        // Invert the function w.r.t. the variable represented by one of
        // the axes. The function values must change monotonously along
        // the chosen axis. Note that this operation is meaningful only
        // in case "Numeric" type is either float or double.
        */
        template <typename ConvertibleToUnsigned>
        CPP11_auto_ptr<LinInterpolatedTableND> invertWRTAxis(
            ConvertibleToUnsigned axisNumber, const Axis& replacementAxis,
            bool newAxisLeftLinear, bool newAxisRightLinear,
            const char* functionLabel=0) const;

        /**
        // This method inverts the ratio response.
        // That is, we assume that the table encodes r(x) for
        // some function f(x) = x*r(x). We also assume that the
        // original axis does not represent x directly -- instead,
        // axis coordinates are given by g(x) (in practice, g is
        // often the natural log). We will also assume that the new
        // axis is not going to represent f(x) directly -- it
        // will be h(f(x)) instead. The functors "invg" and "invh"
        // below must do the inverse: taking the axes coordinates
        // to the actual values of x and f(x). Both "invg" and "invh"
        // must be monotonously increasing functions. The code assumes
        // that x*r(x) -> 0 when x->0 (that is, r(x) is bounded at 0)
        // and it also assumes (but does not check explicitly)
        // that x*r(x) is monotonously increasing with x.
        //
        // The returned interpolation table encodes the values
        // of x/f(x). Of course, they are just 1/r(x), but the trick
        // is to be able to look them up quickly as a function of
        // h(f(x)). Naturally, this whole operation is meaningful
        // only in case "Numeric" type is either float or double.
        */
        template <class Functor1, class Functor2>
        CPP11_auto_ptr<LinInterpolatedTableND> invertRatioResponse(
            unsigned axisNumber, const Axis& replacementAxis,
            bool newAxisLeftLinear, bool newAxisRightLinear,
            Functor1 invg, Functor2 invh,
            const char* functionLabel=0) const;

        /** Comparison for equality */
        bool operator==(const LinInterpolatedTableND&) const;

        /** Logical negation of operator== */
        inline bool operator!=(const LinInterpolatedTableND& r) const
            {return !(*this == r);}

        //@{
        // Method related to "geners" I/O
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static const char* classname();
        static inline unsigned version() {return 1;}
        static LinInterpolatedTableND* read(
            const gs::ClassId& id, std::istream& in);

    private:
        LinInterpolatedTableND();

        LinInterpolatedTableND(
            const ArrayND<Numeric>& data,
            const std::vector<Axis>& axes,
            const char* leftInterpolation,
            const char* rightInterpolation,
            const std::string& label);

        bool allConstInterpolated() const;

        ArrayND<Numeric> data_;
        std::vector<Axis> axes_;
        std::string functionLabel_;
        char leftInterpolationLinear_[CHAR_BIT*sizeof(unsigned long)];
        char rightInterpolationLinear_[CHAR_BIT*sizeof(unsigned long)];
        unsigned dim_;
        bool allConstInterpolated_;

        template <class Functor1>
        static double solveForRatioArg(double xmin, double xmax,
                                       double rmin, double rmax,
                                       double fval, Functor1 invg);

        template <class Functor1>
        static void invert1DResponse(const ArrayND<Numeric>& fromSlice,
                                     const Axis& fromAxis, const Axis& toAxis,
                                     bool newLeftLinear, bool newRightLinear,
                                     Functor1 invg,
                                     const double* rawx, const double* rawf,
                                     double* workspace,
                                     ArrayND<Numeric>* toSlice);
    };
}

#include <cmath>
#include <cfloat>
#include <cassert>
#include <algorithm>
#include <functional>

#include "Alignment/Geners/interface/binaryIO.hh"

#include "JetMETCorrections/InterpolationTables/interface/ArrayNDScanner.h"
#include "JetMETCorrections/InterpolationTables/interface/isMonotonous.h"

namespace npstat {
    namespace Private {
        template <class Axis>
        ArrayShape makeTableShape(const std::vector<Axis>& axes)
        {
            const unsigned n = axes.size();
            ArrayShape result;
            result.reserve(n);
            for (unsigned i=0; i<n; ++i)
                result.push_back(axes[i].nCoords());
            return result;
        }

        template <class Axis>
        ArrayShape makeTableShape(const Axis& xAxis)
        {
            ArrayShape result;
            result.reserve(1U);
            result.push_back(xAxis.nCoords());
            return result;
        }

        template <class Axis>
        ArrayShape makeTableShape(const Axis& xAxis, const Axis& yAxis)
        {
            ArrayShape result;
            result.reserve(2U);
            result.push_back(xAxis.nCoords());
            result.push_back(yAxis.nCoords());
            return result;
        }

        template <class Axis>
        ArrayShape makeTableShape(const Axis& xAxis,
                                  const Axis& yAxis,
                                  const Axis& zAxis)
        {
            ArrayShape result;
            result.reserve(3U);
            result.push_back(xAxis.nCoords());
            result.push_back(yAxis.nCoords());
            result.push_back(zAxis.nCoords());
            return result;
        }

        template <class Axis>
        ArrayShape makeTableShape(const Axis& xAxis, const Axis& yAxis,
                                  const Axis& zAxis, const Axis& tAxis)
        {
            ArrayShape result;
            result.reserve(4U);
            result.push_back(xAxis.nCoords());
            result.push_back(yAxis.nCoords());
            result.push_back(zAxis.nCoords());
            result.push_back(tAxis.nCoords());
            return result;
        }

        template <class Axis>
        ArrayShape makeTableShape(const Axis& xAxis, const Axis& yAxis,
                                  const Axis& zAxis, const Axis& tAxis,
                                  const Axis& vAxis)
        {
            ArrayShape result;
            result.reserve(5U);
            result.push_back(xAxis.nCoords());
            result.push_back(yAxis.nCoords());
            result.push_back(zAxis.nCoords());
            result.push_back(tAxis.nCoords());
            result.push_back(vAxis.nCoords());
            return result;
        }

        inline double lind_interpolateSimple(
            const double x0, const double x1,
            const double y0, const double y1,
            const double x)
        {
            return y0 + (y1 - y0)*((x - x0)/(x1 - x0));
        }

        template <typename Numeric, class Axis>
        void lind_invert1DSlice(
            const ArrayND<Numeric>& fromSlice,
            const Axis& fromAxis, const Axis& toAxis,
            const bool leftLinear, const bool rightLinear,
            ArrayND<Numeric>* toSlice)
        {
            assert(toSlice);
            assert(fromSlice.rank() == 1U);
            assert(toSlice->rank() == 1U);

            const Numeric* fromData = fromSlice.data();
            const unsigned fromLen = fromSlice.length();
            assert(fromLen > 1U);
            assert(fromLen == fromAxis.nCoords());
            const Numeric* fromDataEnd = fromData + fromLen;
            if (!isStrictlyMonotonous(fromData, fromDataEnd))
                throw npstat::NpstatInvalidArgument(
                    "In npstat::Private::lind_invert1DSlice: "
                    "slice data is not monotonous and can not be inverted");

            const Numeric yfirst = fromData[0];
            const Numeric ylast = fromData[fromLen - 1U];
            const bool increasing = yfirst < ylast;

            Numeric* toD = const_cast<Numeric*>(toSlice->data());
            const unsigned nAxisPoints = toAxis.nCoords();
            assert(toSlice->length() == nAxisPoints);

            for (unsigned ipt=0; ipt<nAxisPoints; ++ipt)
            {
                const Numeric y = static_cast<Numeric>(toAxis.coordinate(ipt));
                if (increasing)
                {
                    if (y <= yfirst)
                    {
                        if (leftLinear)
                            toD[ipt] = Private::lind_interpolateSimple(
                                yfirst, fromData[1], fromAxis.coordinate(0),
                                fromAxis.coordinate(1), y);
                        else
                            toD[ipt] = fromAxis.coordinate(0);
                    }
                    else if (y >= ylast)
                    {
                        if (rightLinear)
                            toD[ipt] = Private::lind_interpolateSimple(
                                ylast, fromData[fromLen - 2U],
                                fromAxis.coordinate(fromLen - 1U),
                                fromAxis.coordinate(fromLen - 2U), y);
                        else
                            toD[ipt] = fromAxis.coordinate(fromLen - 1U);
                    }
                    else
                    {
                        const unsigned i = std::lower_bound(fromData,fromDataEnd,y)-
                            fromData;
                        toD[ipt] = Private::lind_interpolateSimple(
                            fromData[i-1U], fromData[i],
                            fromAxis.coordinate(i-1U),
                            fromAxis.coordinate(i), y);
                    }
                }
                else
                {
                    // The role of left and right are exchanged
                    // with respect to first and last point
                    if (y <= ylast)
                    {
                        if (leftLinear)
                            toD[ipt] = Private::lind_interpolateSimple(
                                ylast, fromData[fromLen - 2U],
                                fromAxis.coordinate(fromLen - 1U),
                                fromAxis.coordinate(fromLen - 2U), y);
                        else
                            toD[ipt] = fromAxis.coordinate(fromLen - 1U);
                    }
                    else if (y >= yfirst)
                    {
                        if (rightLinear)
                            toD[ipt] = Private::lind_interpolateSimple(
                                yfirst, fromData[1],
                                fromAxis.coordinate(0),
                                fromAxis.coordinate(1), y);
                        else
                            toD[ipt] = fromAxis.coordinate(0);
                    }
                    else
                    {
                        const unsigned i = std::lower_bound(fromData,fromDataEnd,
                                             y,std::greater<Numeric>())-fromData;
                        toD[ipt] = Private::lind_interpolateSimple(
                            fromData[i-1U], fromData[i],
                            fromAxis.coordinate(i-1U),
                            fromAxis.coordinate(i), y);
                    }
                }
            }
        }
    }

    template <class Numeric, class Axis>
    bool LinInterpolatedTableND<Numeric,Axis>::allConstInterpolated() const
    {
        for (unsigned i=0; i<dim_; ++i)
            if (leftInterpolationLinear_[i] || rightInterpolationLinear_[i])
                return false;
        return true;
    }

    template <class Numeric, class Axis>
    bool LinInterpolatedTableND<Numeric,Axis>::operator==(
        const LinInterpolatedTableND& r) const
    {
        if (dim_ != r.dim_)
            return false;
        for (unsigned i=0; i<dim_; ++i)
        {
            if (leftInterpolationLinear_[i] != r.leftInterpolationLinear_[i])
                return false;
            if (rightInterpolationLinear_[i] != r.rightInterpolationLinear_[i])
                return false;
        }
        return data_ == r.data_ && 
               axes_ == r.axes_ &&
               functionLabel_ == r.functionLabel_;
    }

    template <typename Numeric, class Axis>
    const char* LinInterpolatedTableND<Numeric,Axis>::classname()
    {
        static const std::string myClass(gs::template_class_name<Numeric,Axis>(
                                           "npstat::LinInterpolatedTableND"));
        return myClass.c_str();
    }

    template<typename Numeric, class Axis>
    bool LinInterpolatedTableND<Numeric,Axis>::write(std::ostream& of) const
    {
        const bool status = data_.classId().write(of) &&
                            data_.write(of) &&
                            gs::write_obj_vector(of, axes_);
        if (status)
        {
            gs::write_pod_array(of, leftInterpolationLinear_, dim_);
            gs::write_pod_array(of, rightInterpolationLinear_, dim_);
            gs::write_pod(of, functionLabel_);
        }
        return status && !of.fail();
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>*
    LinInterpolatedTableND<Numeric,Axis>::read(
        const gs::ClassId& id, std::istream& in)
    {
        static const gs::ClassId current(
            gs::ClassId::makeId<LinInterpolatedTableND<Numeric,Axis> >());
        current.ensureSameId(id);

        gs::ClassId ida(in, 1);
        ArrayND<Numeric> data;
        ArrayND<Numeric>::restore(ida, in, &data);
        std::vector<Axis> axes;
        gs::read_heap_obj_vector_as_placed(in, &axes);
        const unsigned dim = axes.size();
        if (dim > CHAR_BIT*sizeof(unsigned long) || data.rank() != dim)
            throw gs::IOInvalidData(
                "In npstat::LinInterpolatedTableND::read: "
                "read back invalid dimensionality");
        char leftInterpolation[CHAR_BIT*sizeof(unsigned long)];
        gs::read_pod_array(in, leftInterpolation, dim);
        char rightInterpolation[CHAR_BIT*sizeof(unsigned long)];
        gs::read_pod_array(in, rightInterpolation, dim);
        std::string label;
        gs::read_pod(in, &label);
        if (in.fail()) throw gs::IOReadFailure(
            "In npstat::LinInterpolatedTableND::read: input stream failure");
        return new LinInterpolatedTableND(
            data, axes, leftInterpolation, rightInterpolation, label);
    }

    template<typename Numeric, class Axis>
    bool LinInterpolatedTableND<Numeric,Axis>::leftInterpolationLinear(
        const unsigned i) const
    {
        if (i >= dim_) throw npstat::NpstatOutOfRange(
            "In npstat::LinInterpolatedTableND::leftInterpolationLinear: "
            "index out of range");
        return leftInterpolationLinear_[i];
    }

    template<typename Numeric, class Axis>
    bool LinInterpolatedTableND<Numeric,Axis>::rightInterpolationLinear(
        const unsigned i) const
    {
        if (i >= dim_) throw npstat::NpstatOutOfRange(
            "In npstat::LinInterpolatedTableND::rightInterpolationLinear: "
            "index out of range");
        return rightInterpolationLinear_[i];
    }

    template<typename Numeric, class Axis>
    bool LinInterpolatedTableND<Numeric,Axis>::isUniformlyBinned() const
    {
        for (unsigned i=0; i<dim_; ++i)
            if (!axes_[i].isUniform())
                return false;
        return true;
    }

    template<typename Numeric, class Axis>
    std::vector<std::pair<bool,bool> >
    LinInterpolatedTableND<Numeric,Axis>::interpolationType() const
    {
        std::vector<std::pair<bool,bool> > vec;
        vec.reserve(dim_);
        for (unsigned i=0; i<dim_; ++i)
            vec.push_back(std::pair<bool, bool>(leftInterpolationLinear_[i],
                                                rightInterpolationLinear_[i]));
        return vec;
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const std::vector<Axis>& axes,
        const std::vector<std::pair<bool,bool> >& interpolationType,
        const char* label)
        : data_(Private::makeTableShape(axes)),
          axes_(axes),
          functionLabel_(label ? label : ""),
          dim_(axes.size())
    {
        if (dim_ == 0 || dim_ >= CHAR_BIT*sizeof(unsigned long))
            throw npstat::NpstatInvalidArgument(
                "In npstat::LinInterpolatedTableND constructor: requested "
                "table dimensionality is not supported");
        if (dim_ != interpolationType.size())
            throw npstat::NpstatInvalidArgument(
                "In npstat::LinInterpolatedTableND constructor: "
                "incompatible number of interpolation specifications");
        for (unsigned i=0; i<dim_; ++i)
        {
            const std::pair<bool,bool>& pair(interpolationType[i]);
            leftInterpolationLinear_[i] = pair.first;
            rightInterpolationLinear_[i] = pair.second;
        }

        allConstInterpolated_ = allConstInterpolated();
    }

    template<typename Numeric, class Axis>
    template <class Num2>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const LinInterpolatedTableND<Num2,Axis>& r)
        : data_(r.data_),
          axes_(r.axes_),
          functionLabel_(r.functionLabel_),
          dim_(r.dim_),
          allConstInterpolated_(r.allConstInterpolated_)
    {
        for (unsigned i=0; i<dim_; ++i)
        {
            leftInterpolationLinear_[i] = r.leftInterpolationLinear_[i];
            rightInterpolationLinear_[i] = r.rightInterpolationLinear_[i];
        }
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const ArrayND<Numeric>& data,
        const std::vector<Axis>& axes,
        const char* leftInterpolation,
        const char* rightInterpolation,
        const std::string& label)
        : data_(data),
          axes_(axes),
          functionLabel_(label),
          dim_(data.rank())
    {
        for (unsigned i=0; i<dim_; ++i)
        {
            leftInterpolationLinear_[i] = leftInterpolation[i];
            rightInterpolationLinear_[i] = rightInterpolation[i];
        }
        allConstInterpolated_ = allConstInterpolated();
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const Axis& xAxis, bool leftX, bool rightX,
        const Axis& yAxis, bool leftY, bool rightY,
        const Axis& zAxis, bool leftZ, bool rightZ,
        const Axis& tAxis, bool leftT, bool rightT,
        const Axis& vAxis, bool leftV, bool rightV,
        const char* label)
        : data_(Private::makeTableShape(xAxis, yAxis, zAxis, tAxis, vAxis)),
          functionLabel_(label ? label : ""),
          dim_(5U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);
        axes_.push_back(zAxis);
        axes_.push_back(tAxis);
        axes_.push_back(vAxis);

        unsigned i = 0;
        leftInterpolationLinear_[i] = leftX;
        rightInterpolationLinear_[i++] = rightX;
        leftInterpolationLinear_[i] = leftY;
        rightInterpolationLinear_[i++] = rightY;
        leftInterpolationLinear_[i] = leftZ;
        rightInterpolationLinear_[i++] = rightZ;
        leftInterpolationLinear_[i] = leftT;
        rightInterpolationLinear_[i++] = rightT;
        leftInterpolationLinear_[i] = leftV;
        rightInterpolationLinear_[i++] = rightV;
        assert(i == dim_);

        allConstInterpolated_ = allConstInterpolated();
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const Axis& xAxis, bool leftX, bool rightX,
        const Axis& yAxis, bool leftY, bool rightY,
        const Axis& zAxis, bool leftZ, bool rightZ,
        const Axis& tAxis, bool leftT, bool rightT,
        const char* label)
        : data_(Private::makeTableShape(xAxis, yAxis, zAxis, tAxis)),
          functionLabel_(label ? label : ""),
          dim_(4U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);
        axes_.push_back(zAxis);
        axes_.push_back(tAxis);

        unsigned i = 0;
        leftInterpolationLinear_[i] = leftX;
        rightInterpolationLinear_[i++] = rightX;
        leftInterpolationLinear_[i] = leftY;
        rightInterpolationLinear_[i++] = rightY;
        leftInterpolationLinear_[i] = leftZ;
        rightInterpolationLinear_[i++] = rightZ;
        leftInterpolationLinear_[i] = leftT;
        rightInterpolationLinear_[i++] = rightT;
        assert(i == dim_);

        allConstInterpolated_ = allConstInterpolated();
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const Axis& xAxis, bool leftX, bool rightX,
        const Axis& yAxis, bool leftY, bool rightY,
        const Axis& zAxis, bool leftZ, bool rightZ,
        const char* label)
        : data_(Private::makeTableShape(xAxis, yAxis, zAxis)),
          functionLabel_(label ? label : ""),
          dim_(3U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);
        axes_.push_back(zAxis);

        unsigned i = 0;
        leftInterpolationLinear_[i] = leftX;
        rightInterpolationLinear_[i++] = rightX;
        leftInterpolationLinear_[i] = leftY;
        rightInterpolationLinear_[i++] = rightY;
        leftInterpolationLinear_[i] = leftZ;
        rightInterpolationLinear_[i++] = rightZ;
        assert(i == dim_);

        allConstInterpolated_ = allConstInterpolated();
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const Axis& xAxis, bool leftX, bool rightX,
        const Axis& yAxis, bool leftY, bool rightY,
        const char* label)
        : data_(Private::makeTableShape(xAxis, yAxis)),
          functionLabel_(label ? label : ""),
          dim_(2U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);
        axes_.push_back(yAxis);

        unsigned i = 0;
        leftInterpolationLinear_[i] = leftX;
        rightInterpolationLinear_[i++] = rightX;
        leftInterpolationLinear_[i] = leftY;
        rightInterpolationLinear_[i++] = rightY;
        assert(i == dim_);

        allConstInterpolated_ = allConstInterpolated();
    }

    template<typename Numeric, class Axis>
    LinInterpolatedTableND<Numeric,Axis>::LinInterpolatedTableND(
        const Axis& xAxis, bool leftX, bool rightX,
        const char* label)
        : data_(Private::makeTableShape(xAxis)),
          functionLabel_(label ? label : ""),
          dim_(1U)
    {
        axes_.reserve(dim_);
        axes_.push_back(xAxis);

        leftInterpolationLinear_[0] = leftX;
        rightInterpolationLinear_[0] = rightX;

        allConstInterpolated_ = allConstInterpolated();
    }

    template<typename Numeric, class Axis>
    template <typename ConvertibleToUnsigned>
    CPP11_auto_ptr<LinInterpolatedTableND<Numeric,Axis> > 
    LinInterpolatedTableND<Numeric,Axis>::invertWRTAxis(
        const ConvertibleToUnsigned axisNumC, const Axis& replacementAxis,
        const bool leftLinear, const bool rightLinear,
        const char* functionLabel) const
    {
        const unsigned axisNumber = static_cast<unsigned>(axisNumC);

        if (axisNumber >= dim_)
            throw npstat::NpstatOutOfRange(
                "In npstat::LinInterpolatedTableND::invertAxis: "
                "axis number is out of range");

        // Generate the new set of axes
        std::vector<Axis> newAxes(axes_);
        newAxes[axisNumber] = replacementAxis;

        std::vector<std::pair<bool,bool> > iType(interpolationType());
        iType[axisNumber] = std::pair<bool,bool>(leftLinear, rightLinear);

        // Create the new table
        CPP11_auto_ptr<LinInterpolatedTableND> pTable(
            new LinInterpolatedTableND(newAxes, iType, functionLabel));

        if (dim_ > 1U)
        {
            // Prepare array slices
            unsigned sliceIndex[CHAR_BIT*sizeof(unsigned long)];
            unsigned fixedIndices[CHAR_BIT*sizeof(unsigned long)];
            unsigned count = 0;
            for (unsigned i=0; i<dim_; ++i)
                if (i != axisNumber)
                {
                    sliceIndex[count] = data_.span(i);
                    fixedIndices[count++] = i;
                }
            ArrayND<Numeric> parentSlice(data_, fixedIndices, count);
            ArrayND<Numeric> dauSlice(pTable->data_, fixedIndices, count);

            // Cycle over the slices
            for (ArrayNDScanner scan(sliceIndex,count); scan.isValid(); ++scan)
            {
                scan.getIndex(sliceIndex, count);
                data_.exportSlice(&parentSlice, fixedIndices,
                                  sliceIndex, count);
                Private::lind_invert1DSlice(
                    parentSlice, axes_[axisNumber], replacementAxis,
                    leftLinear, rightLinear, &dauSlice);
                pTable->data_.importSlice(dauSlice, fixedIndices,
                                          sliceIndex, count);
            }
        }
        else
            Private::lind_invert1DSlice(
                data_, axes_[0], replacementAxis,
                leftLinear, rightLinear, &pTable->data_);
        return pTable;
    }

    template<typename Numeric, class Axis>
    template <class Functor1, class Functor2>
    CPP11_auto_ptr<LinInterpolatedTableND<Numeric,Axis> > 
    LinInterpolatedTableND<Numeric,Axis>::invertRatioResponse(
        const unsigned axisNumber, const Axis& replacementAxis,
        const bool leftLinear, const bool rightLinear,
        Functor1 invg, Functor2 invh,
        const char* functionLabel) const
    {
        if (axisNumber >= dim_)
            throw npstat::NpstatOutOfRange(
                "In npstat::LinInterpolatedTableND::invertRatioResponse: "
                "axis number is out of range");

        // Generate the new set of axes
        std::vector<Axis> newAxes(axes_);
        newAxes[axisNumber] = replacementAxis;

        std::vector<std::pair<bool,bool> > iType(interpolationType());
        iType[axisNumber] = std::pair<bool,bool>(leftLinear, rightLinear);

        // Transform the original axis to the raw x values
        const Axis& oldAxis(axes_[axisNumber]);
        std::vector<double> rawx;
        const unsigned nCoords = oldAxis.nCoords();
        rawx.reserve(nCoords);
        for (unsigned i=0; i<nCoords; ++i)
        {
            const double x = invg(oldAxis.coordinate(i));
            if (x < 0.0)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::LinInterpolatedTableND::invertRatioResponse: "
                    "invalid original axis definition (negative transformed "
                    "coordinate)");
            rawx.push_back(x);
        }

        // Transform the new axis to the raw f(x) values
        std::vector<double> rawf;
        const unsigned nFuncs = replacementAxis.nCoords();
        rawf.reserve(nFuncs);
        for (unsigned i=0; i<nFuncs; ++i)
        {
            const double f = invh(replacementAxis.coordinate(i));
            if (f < 0.0)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::LinInterpolatedTableND::invertRatioResponse: "
                    "invalid new axis definition (negative transformed "
                    "coordinate)");
            rawf.push_back(f);
        }

        // Workspace needed for the inversion code
        std::vector<double> workspace(nCoords);

        // Create the new table
        CPP11_auto_ptr<LinInterpolatedTableND> pTable(
            new LinInterpolatedTableND(newAxes, iType, functionLabel));

        if (dim_ > 1U)
        {
            // Prepare array slices
            unsigned sliceIndex[CHAR_BIT*sizeof(unsigned long)];
            unsigned fixedIndices[CHAR_BIT*sizeof(unsigned long)];
            unsigned count = 0;
            for (unsigned i=0; i<dim_; ++i)
                if (i != axisNumber)
                {
                    sliceIndex[count] = data_.span(i);
                    fixedIndices[count++] = i;
                }
            ArrayND<Numeric> parentSlice(data_, fixedIndices, count);
            ArrayND<Numeric> dauSlice(pTable->data_, fixedIndices, count);

            // Cycle over the slices
            for (ArrayNDScanner scan(sliceIndex,count); scan.isValid(); ++scan)
            {
                scan.getIndex(sliceIndex, count);
                data_.exportSlice(&parentSlice, fixedIndices,
                                  sliceIndex, count);
                invert1DResponse(parentSlice, oldAxis,
                                 replacementAxis, leftLinear, rightLinear,
                                 invg, &rawx[0], &rawf[0], &workspace[0],
                                 &dauSlice);
                pTable->data_.importSlice(dauSlice, fixedIndices,
                                          sliceIndex, count);
            }
        }
        else
            invert1DResponse(data_, oldAxis, replacementAxis, leftLinear,
                             rightLinear, invg, &rawx[0], &rawf[0],
                             &workspace[0], &pTable->data_);
        return pTable;
    }

    template<typename Numeric, class Axis>
    void LinInterpolatedTableND<Numeric,Axis>::getCoords(
        const unsigned long linearIndex,
        double* coords, const unsigned coordsBufferSize) const
    {
        if (coordsBufferSize < dim_) throw npstat::NpstatInvalidArgument(
            "In LinInterpolatedTableND::getCoords: "
            "insufficient buffer size");
        assert(coords);
        unsigned index[CHAR_BIT*sizeof(unsigned long)];
        data_.convertLinearIndex(linearIndex, index, dim_);
        for (unsigned i=0; i<dim_; ++i)
            coords[i] = axes_[i].coordinate(index[i]);
    }

    template<typename Numeric, class Axis>
    bool LinInterpolatedTableND<Numeric,Axis>::isWithinLimits(
        const double* point, const unsigned len) const
    {
        if (len != dim_)
            throw npstat::NpstatInvalidArgument(
                "In npstat::LinInterpolatedTableND::isWithinLimits: "
                "incompatible point dimensionality");
        assert(point);

        for (unsigned i=0; i<dim_; ++i)
            if (point[i] < axes_[i].min() || point[i] > axes_[i].max())
                return false;
        return true;
    }

    template<typename Numeric, class Axis>
    Numeric LinInterpolatedTableND<Numeric,Axis>::operator()(
        const double* point, const unsigned len) const
    {
        typedef typename ProperDblFromCmpl<Numeric>::type proper_double;

        if (len != dim_)
            throw npstat::NpstatInvalidArgument(
                "In npstat::LinInterpolatedTableND::operator(): "
                "incompatible point dimensionality");
        assert(point);

        bool interpolateArray = true;
        if (!allConstInterpolated_)
            for (unsigned i=0; i<dim_; ++i)
                if ((leftInterpolationLinear_[i] && point[i] < axes_[i].min()) ||
                    (rightInterpolationLinear_[i] && point[i] > axes_[i].max()))
                {
                    interpolateArray = false;
                    break;
                }

        if (interpolateArray)
        {
            // Translate coordinates into the array system and
            // simply use the ArrayND interpolation facilities
            double buf[CHAR_BIT*sizeof(unsigned long)];
            for (unsigned i=0; i<dim_; ++i)
            {
                const std::pair<unsigned,double>& pair = 
                    axes_[i].getInterval(point[i]);
                buf[i] = pair.first + 1U - pair.second;
            }
            return data_.interpolate1(buf, dim_);
        }
        else
        {
            unsigned ix[CHAR_BIT*sizeof(unsigned long)];
            double weight[CHAR_BIT*sizeof(unsigned long)];
            for (unsigned i=0; i<dim_; ++i)
            {
                const bool linear = (leftInterpolationLinear_[i] && 
                                     point[i] < axes_[i].min()) ||
                                    (rightInterpolationLinear_[i] && 
                                     point[i] > axes_[i].max());
                const std::pair<unsigned,double>& pair = linear ?
                    axes_[i].linearInterval(point[i]) :
                    axes_[i].getInterval(point[i]);
                ix[i] = pair.first;
                weight[i] = pair.second;
            }

            Numeric sum = Numeric();
            const unsigned long maxcycle = 1UL << dim_;
            const unsigned long* strides = data_.strides();
            const Numeric* dat = data_.data();
            for (unsigned long icycle=0UL; icycle<maxcycle; ++icycle)
            {
                double w = 1.0;
                unsigned long icell = 0UL;
                for (unsigned i=0; i<dim_; ++i)
                {
                    if (icycle & (1UL << i))
                    {
                        w *= (1.0 - weight[i]);
                        icell += strides[i]*(ix[i] + 1U);
                    }
                    else
                    {
                        w *= weight[i];
                        icell += strides[i]*ix[i];
                    }
                }
                sum += dat[icell]*static_cast<proper_double>(w);
            }
            return sum;
        }
    }

    template<typename Numeric, class Axis>
    Numeric LinInterpolatedTableND<Numeric,Axis>::operator()(
        const double& x0) const
    {
        const unsigned nArgs = 1U;
        if (dim_ != nArgs) throw npstat::NpstatInvalidArgument(
            "In npstat::LinInterpolatedTableND::operator(): number of "
            "arguments, 1, is incompatible with the interpolator dimensionality");
        double tmp[nArgs];
        tmp[0] = x0;
        return operator()(tmp, nArgs);
    }

    template<typename Numeric, class Axis>
    Numeric LinInterpolatedTableND<Numeric,Axis>::operator()(
        const double& x0, const double& x1) const
    {
        const unsigned nArgs = 2U;
        if (dim_ != nArgs) throw npstat::NpstatInvalidArgument(
            "In npstat::LinInterpolatedTableND::operator(): number of "
            "arguments, 2, is incompatible with the interpolator dimensionality");
        double tmp[nArgs];
        tmp[0] = x0;
        tmp[1] = x1;
        return operator()(tmp, nArgs);
    }

    template<typename Numeric, class Axis>
    Numeric LinInterpolatedTableND<Numeric,Axis>::operator()(
        const double& x0, const double& x1, const double& x2) const
    {
        const unsigned nArgs = 3U;
        if (dim_ != nArgs) throw npstat::NpstatInvalidArgument(
            "In npstat::LinInterpolatedTableND::operator(): number of "
            "arguments, 3, is incompatible with the interpolator dimensionality");
        double tmp[nArgs];
        tmp[0] = x0;
        tmp[1] = x1;
        tmp[2] = x2;
        return operator()(tmp, nArgs);
    }

    template<typename Numeric, class Axis>
    Numeric LinInterpolatedTableND<Numeric,Axis>::operator()(
        const double& x0, const double& x1,
        const double& x2, const double& x3) const
    {
        const unsigned nArgs = 4U;
        if (dim_ != nArgs) throw npstat::NpstatInvalidArgument(
            "In npstat::LinInterpolatedTableND::operator(): number of "
            "arguments, 4, is incompatible with the interpolator dimensionality");
        double tmp[nArgs];
        tmp[0] = x0;
        tmp[1] = x1;
        tmp[2] = x2;
        tmp[3] = x3;
        return operator()(tmp, nArgs);
    }

    template<typename Numeric, class Axis>
    Numeric LinInterpolatedTableND<Numeric,Axis>::operator()(
        const double& x0, const double& x1, const double& x2,
        const double& x3, const double& x4) const
    {
        const unsigned nArgs = 5U;
        if (dim_ != nArgs) throw npstat::NpstatInvalidArgument(
            "In npstat::LinInterpolatedTableND::operator(): number of "
            "arguments, 5, is incompatible with the interpolator dimensionality");
        double tmp[nArgs];
        tmp[0] = x0;
        tmp[1] = x1;
        tmp[2] = x2;
        tmp[3] = x3;
        tmp[4] = x4;
        return operator()(tmp, nArgs);
    }

    template<typename Numeric, class Axis>
    template <class Functor1>
    double LinInterpolatedTableND<Numeric,Axis>::solveForRatioArg(
        const double xmin, const double xmax,
        const double rmin, const double rmax,
        const double fval, Functor1 invg)
    {
        // Find two values of x so that f(x0) <= fval <= f(x1)
        double x0 = xmin;
        double x1 = xmax;
        double fmin = invg(xmin)*rmin;
        double fmax = invg(xmax)*rmax;
        const double step = xmax - xmin;
        assert(fmin < fmax);
        assert(step > 0.0);

        unsigned stepcount = 0;
        const unsigned maxSteps = 1000U;
        for (double stepfactor = 1.0; (fval < fmin || fval > fmax) &&
                 stepcount < maxSteps; stepfactor *= 2.0, ++stepcount)
            if (fval < fmin)
            {
                x1 = x0;
                fmax = fmin;
                x0 -= stepfactor*step;
                fmin = invg(x0)*Private::lind_interpolateSimple(
                    xmin, xmax, rmin, rmax, x0);
            }
            else
            {
                x0 = x1;
                fmin = fmax;
                x1 += stepfactor*step;
                fmax = invg(x1)*Private::lind_interpolateSimple(
                    xmin, xmax, rmin, rmax, x1);
            }
        if (stepcount == maxSteps) throw npstat::NpstatRuntimeError(
            "In LinInterpolatedTableND::solveForRatioArg: "
            "faled to bracket the root");

        assert(x1 >= x0);
        while ((x1 - x0)/(std::abs(x1) + std::abs(x0) + DBL_EPSILON) > 4.0*DBL_EPSILON)
        {
            const double xhalf = (x1 + x0)/2.0;
            const double fhalf = invg(xhalf)*Private::lind_interpolateSimple(
                xmin, xmax, rmin, rmax, xhalf);
            if (fval < fhalf)
            {
                x1 = xhalf;
                fmax = fhalf;
            }
            else
            {
                x0 = xhalf;
                fmin = fhalf;
            }    
        }
        return (x1 + x0)/2.0;
    }

    template<typename Numeric, class Axis>
    template <class Functor1>
    void LinInterpolatedTableND<Numeric,Axis>::invert1DResponse(
        const ArrayND<Numeric>& fromSlice,
        const Axis& fromAxis, const Axis& toAxis,
        const bool newLeftLinear, const bool newRightLinear,
        Functor1 invg, const double* rawx, const double* rawf,
        double* workspace,
        ArrayND<Numeric>* toSlice)
    {
        assert(toSlice);
        assert(fromSlice.rank() == 1U);
        assert(toSlice->rank() == 1U);

        const Numeric zero = Numeric();
        const Numeric* fromData = fromSlice.data();
        const unsigned fromLen = fromSlice.length();
        assert(fromLen > 1U);
        assert(fromLen == fromAxis.nCoords());
        Numeric* toD = const_cast<Numeric*>(toSlice->data());
        const unsigned nAxisPoints = toAxis.nCoords();
        assert(toSlice->length() == nAxisPoints);

        for (unsigned i=0; i<fromLen; ++i)
        {
            if (fromData[i] <= zero) throw npstat::NpstatDomainError(
                "In LinInterpolatedTableND::invert1DResponse: "
                "non-positive response found. This ratio "
                "response table is not invertible.");
            workspace[i] = rawx[i]*fromData[i];
        }

        const double yfirst = workspace[0];
        const double ylast = workspace[fromLen - 1U];

        bool adjustZero = false;
        unsigned nBelow = 0;
        for (unsigned ipt=0; ipt<nAxisPoints; ++ipt)
        {
            const double y = rawf[ipt];
            unsigned i0 = 0;
            bool solve = false;
            if (y == 0.0)
            {
                assert(ipt == 0U);
                if (newLeftLinear)
                    adjustZero = true;
            }
            else if (y <= yfirst)
            {
                ++nBelow;
                solve = newLeftLinear;
            }
            else if (y >= ylast)
            {
                solve = newRightLinear;
                i0 = solve ? fromLen-2 : fromLen-1;
            }
            else
            {
                solve = true;
                i0 = static_cast<unsigned>(std::lower_bound(
                         workspace,workspace+fromLen,y) - workspace) - 1U;
            }
            if (solve)
            {
                const double x = solveForRatioArg(fromAxis.coordinate(i0),
                                                  fromAxis.coordinate(i0+1),
                                                  fromData[i0], fromData[i0+1],
                                                  y, invg);
                toD[ipt] = invg(x)/y;
            }
            else
                toD[ipt] = 1.0/fromData[i0];
        }
        if (adjustZero && nBelow)
            toD[0] = toD[1];
    }
}


#endif // NPSTAT_LININTERPOLATEDTABLEND_HH_

