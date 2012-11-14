#ifndef NPSTAT_UNIFORMAXIS_HH_
#define NPSTAT_UNIFORMAXIS_HH_

/*!
// \file UniformAxis.h
//
// \brief Uniformly spaced coordinate sets for use in constructing
//        rectangular grids
//
// Author: I. Volobouev
//
// June 2012
*/

#include <vector>
#include <utility>
#include <string>
#include <iostream>

#include "Alignment/Geners/interface/ClassId.hh"

namespace npstat {
    /**
    // This class contains the info needed to define an axis of a rectangular
    // grid. The distance between grid points is uniform.
    */
    class UniformAxis
    {
    public:
        // The number of coordinates must be at least 2
        UniformAxis(unsigned nCoords, double min, double max,
                    const char* label=0);

        // Basic accessors
        inline unsigned nCoords() const {return npt_;}
        inline double min() const {return min_;}
        inline double max() const {return max_;}
        inline const std::string& label() const {return label_;}
        inline bool usesLogSpace() const {return false;}

        // The following function returns the grid interval number and
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
        std::pair<unsigned,double> getInterval(double coordinate) const;

        // Similar function which calculates the weights including
        // the points outside of the axis boundaries
        std::pair<unsigned,double> linearInterval(double coordinate) const;

        // Convenience methods
        std::vector<double> coords() const;
        double coordinate(unsigned i) const;
        inline double length() const {return max_ - min_;}
        inline bool isUniform() const {return true;}
        inline unsigned nIntervals() const {return npt_ - 1;}
        inline double intervalWidth(unsigned) const {return bw_;}

        bool operator==(const UniformAxis& r) const;
        inline bool operator!=(const UniformAxis& r) const
            {return !(*this == r);}

        // Closeness within tolerance
        bool isClose(const UniformAxis& r, double tol) const;

        // Modify the label
        inline void setLabel(const char* newlabel)
            {label_ = newlabel ? newlabel : "";}

        // Methods related to I/O
        inline gs::ClassId classId() const {return gs::ClassId(*this);}
        bool write(std::ostream& of) const;

        static inline const char* classname() {return "npstat::UniformAxis";}
        static inline unsigned version() {return 1;}
        static UniformAxis* read(const gs::ClassId& id, std::istream& in);

    private:
        inline UniformAxis() : min_(0.), max_(0.), bw_(0.), npt_(0) {}

        double min_;
        double max_;
        double bw_;
        std::string label_;
        unsigned npt_;

    };
}

#endif // NPSTAT_UNIFORMAXIS_HH_

