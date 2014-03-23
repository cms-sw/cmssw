#ifndef NPSTAT_GRIDAXIS_HH_
#define NPSTAT_GRIDAXIS_HH_

/*!
// \file GridAxis.h
//
// \brief Non-uniformly spaced coordinate sets for use in constructing
//        rectangular grids
//
// Author: I. Volobouev
//
// July 2010
*/

#include <vector>
#include <utility>
#include <string>
#include <iostream>

#include "Alignment/Geners/interface/ClassId.hh"

namespace npstat {
    /**
    // Information needed to define an axis of a rectangular grid.
    // The distance between grid points can change from point to point.
    //
    // The UniformAxis class will be more efficient in representing
    // equidistant grids.
    */
    class GridAxis
    {
    public:
        //@{
        /**
        // The number of grid coordinates provided must be at least 2.
        // Coordinates will be sorted internally in the increasing order.
        */
        explicit GridAxis(const std::vector<double>& coords,
                          bool useLogSpace=false);
        GridAxis(const std::vector<double>& coords, const char* label,
                 bool useLogSpace=false);
        //@}

        //@{
        /** Basic accessor returning a parameter provided in the constructor */
        inline const std::vector<double>& coords() const {return coords_;}
        inline const std::string& label() const {return label_;}
        inline bool usesLogSpace() const {return useLogSpace_;}
        //@}

        /**
        // This method returns the grid interval number and
        // the weight of the point at the left side of the interval.
        // The weight will be set to 1 if the given coordinate coincides
        // with the grid point and will decay to 0 linearly as the
        // coordinate moves towards the next point on the right.
        //
        // The coordinates below the leftmost grid point are mapped
        // into the 0th interval with weight 1. The coordinates above
        // the rightmost grid point are mapped into the last interval
        // with weight 0 for the left point (it is expected that weight 1
        // will then be assigned to the right point).
        */
        std::pair<unsigned,double> getInterval(double coordinate) const;

        /**
        // This method returns the grid interval number and
        // the weight of the point at the left side of the interval.
        // The weight will be set to 1 if the given coordinate coincides
        // with the grid point and will decay to 0 linearly as the
        // coordinate moves towards the next point on the right.
        // The weight for the point on the right should be set to
        // one minus the weight on the left.
        //
        // The coordinates outside of grid boundaries will result in
        // weights which are less than zero or more than one. They
        // will be calculated by linear extrapolation from the closest
        // interval in the grid (i.e., leftmost or rightmost).
        */
        std::pair<unsigned,double> linearInterval(double coordinate) const;

        //@{
        /** Convenience accessor */
        inline unsigned nCoords() const {return npt_;}
        inline double coordinate(const unsigned i) const
            {return coords_.at(i);}
        inline double min() const {return coords_.front();}
        inline double max() const {return coords_.back();}
        inline double length() const {return coords_.back() - coords_.front();}
        inline bool isUniform() const {return false;}
        inline unsigned nIntervals() const {return coords_.size() - 1;}
        inline double intervalWidth(const unsigned i=0) const
            {return coords_.at(i+1) - coords_.at(i);}
        //@}

        /** Compare two grids for equality */
        bool operator==(const GridAxis& r) const;

        /** Logical negation of operator== */
        inline bool operator!=(const GridAxis& r) const
            {return !(*this == r);}

        /**
        // Check for closeness of coordinates with another axis
        // within the given relative tolerance
        */
        bool isClose(const GridAxis& r, double tol) const;

        /** Modify the axis label */
        inline void setLabel(const char* newlabel)
            {label_ = newlabel ? newlabel : "";}

        //@{
        /** Method related to "geners" I/O */
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;
        //@}

        static inline const char* classname() {return "npstat::GridAxis";}
        static inline unsigned version() {return 2;}
        static GridAxis* read(const gs::ClassId& id, std::istream& in);

    private:
        void initialize();

        std::vector<double> coords_;
        std::vector<double> logs_;
        std::string label_;
        unsigned npt_;
        bool useLogSpace_;

        inline GridAxis() : npt_(0), useLogSpace_(false) {}
    };
}

#endif // NPSTAT_GRIDAXIS_HH_

