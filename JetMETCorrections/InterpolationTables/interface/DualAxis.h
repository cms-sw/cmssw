#ifndef NPSTAT_DUALAXIS_HH_
#define NPSTAT_DUALAXIS_HH_

/*!
// \file DualAxis.h
//
// \brief Represent both equidistant and non-uniform coordinate sets
//        for rectangular grids
//
// Author: I. Volobouev
//
// July 2012
*/

#include "JetMETCorrections/InterpolationTables/interface/GridAxis.h"
#include "JetMETCorrections/InterpolationTables/interface/UniformAxis.h"

namespace npstat {
    /**
    // Rectangular grid axis which can be either uniform or non-uniform.
    // Will work a little bit slower than either GridAxis or UniformAxis,
    // but can be used in place of either one of them.
    */
    class DualAxis
    {
    public:
        // Constructors
        inline DualAxis(const GridAxis& g)
            : a_(g), u_(2, 0.0, 1.0), uniform_(false) {}

        inline DualAxis(const UniformAxis& u)
            : a_(dummy_vec()), u_(u), uniform_(true) {}

        inline DualAxis(unsigned nCoords, double min, double max,
                        const char* label=0)
            : a_(dummy_vec()), u_(nCoords, min, max, label), uniform_(true) {}

        inline explicit DualAxis(const std::vector<double>& coords,
                                 const bool useLogSpace=false)
            : a_(coords, useLogSpace), u_(2, 0.0, 1.0), uniform_(false) {}

        inline DualAxis(const std::vector<double>& coords, const char* label,
                        const bool useLogSpace=false)
            : a_(coords, label, useLogSpace), u_(2,0.0,1.0), uniform_(false) {}

        // Inspectors
        inline bool isUniform() const {return uniform_;}

        inline unsigned nCoords() const
            {return uniform_ ? u_.nCoords() : a_.nCoords();}

        inline double min() const
            {return uniform_ ? u_.min() : a_.min();}

        inline double max() const
            {return uniform_ ? u_.max() : a_.max();}

        inline const std::string& label() const
            {return uniform_ ? u_.label() : a_.label();}

        inline bool usesLogSpace() const
            {return uniform_ ? u_.usesLogSpace() : a_.usesLogSpace();}

        inline std::pair<unsigned,double> getInterval(const double x) const
            {return uniform_ ? u_.getInterval(x) : a_.getInterval(x);}

        inline std::pair<unsigned,double> linearInterval(const double x) const
            {return uniform_ ? u_.linearInterval(x) : a_.linearInterval(x);}

        inline double coordinate(const unsigned i) const
            {return uniform_ ? u_.coordinate(i) : a_.coordinate(i);}

        inline double length() const
            {return uniform_ ? u_.length() : a_.length();}

        inline unsigned nIntervals() const
            {return uniform_ ? u_.nIntervals() : a_.nIntervals();}

        inline double intervalWidth(const unsigned i=0) const
            {return uniform_ ? u_.intervalWidth(i) : a_.intervalWidth(i);}

        inline std::vector<double> coords() const
            {return uniform_ ? u_.coords() : a_.coords();}

        inline bool operator==(const DualAxis& r) const
            {return uniform_ == r.uniform_ && a_ == r.a_ && u_ == r.u_;}

        inline bool operator!=(const DualAxis& r) const
            {return !(*this == r);}

        //@{
        /**
        // Return a pointer to the underlying axis. This will be
        // a null pointer if the axis does not correspond to the
        // constructed type.
        */
        inline const GridAxis* getGridAxis() const
            {return uniform_ ? static_cast<const GridAxis*>(0) : &a_;}

        inline const UniformAxis* getUniformAxis() const
            {return uniform_ ? &u_ : static_cast<const UniformAxis*>(0);}
        //@}

        /** Modify the axis label */
        inline void setLabel(const char* newlabel)
            {uniform_ ? u_.setLabel(newlabel) : a_.setLabel(newlabel);}

        //@{
        /** Method related to "geners" I/O */
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static inline const char* classname() {return "npstat::DualAxis";}
        static inline unsigned version() {return 1;}
        static DualAxis* read(const gs::ClassId& id, std::istream& in);

    private:
        GridAxis a_;
        UniformAxis u_;
        bool uniform_;

        inline static std::vector<double> dummy_vec()
        {
            std::vector<double> vec(2, 0.0);
            vec[1] = 1.0;
            return vec;
        }

        inline DualAxis()
            : a_(dummy_vec()), u_(2, 0.0, 1.0), uniform_(true) {}
    };
}

#endif // NPSTAT_DUALAXIS_HH_

