#ifndef NPSTAT_INTERPOLATEHISTOND_HH_
#define NPSTAT_INTERPOLATEHISTOND_HH_

/*!
// \file interpolateHistoND.h
//
// \brief Interpolate histogram contents
//
// Functions which interpolate histogram contents are not included
// into the HistoND template itself because we do not always want to
// create histograms using bin types which can be multiplied by doubles
// (also, results of such a multiplication have to be automatically
// converted back to the same type).
//
// The implementations work by invoking "interpolate1" or "interpolate3"
// ArrayND methods on the histogram bin contents after an appropriate
// coordinate transformation.
//
// Author: I. Volobouev
//
// November 2011
*/

#include "JetMETCorrections/InterpolationTables/interface/HistoND.h"

namespace npstat {
    /**
    // The interpolation degree in this method can be set to 0, 1, or 3
    // which results, respectively, in closest bin lookup, multilinear
    // interpolation, or multicubic interpolation. Value of the closest
    // bin inside the histogram range is used if some coordinate is outside
    // of the corresponding axis limits.
    */
    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double *coords, unsigned coordsDim,
                             unsigned interpolationDegree);
    //@{
    /**
    // Convenience function for interpolating histograms, with
    // an explicit coordinate argument for each histogram dimension
    */
    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1,
                             unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2,
                             unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2, double x3,
                             unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2, double x3,
                             double x4, unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2, double x3,
                             double x4, double x5,
                             unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2, double x3,
                             double x4, double x5, double x6,
                             unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2, double x3,
                             double x4, double x5, double x6, double x7,
                             unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2, double x3,
                             double x4, double x5, double x6, double x7,
                             double x8, unsigned interpolationDegree);

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             double x0, double x1, double x2, double x3,
                             double x4, double x5, double x6, double x7,
                             double x8, double x9,
                             unsigned interpolationDegree);
    //@}
}

#include <cassert>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

namespace npstat {
    namespace Private {
        template <typename Float, class Axis>
        void iHND_checkArgs(const HistoND<Float,Axis>& histo,
                            const unsigned xDim,
                            const unsigned interpolationDegree)
        {
            if (xDim != histo.dim()) throw npstat::NpstatInvalidArgument(
                "In npstat::interpolateHistoND: incompatible "
                "dimensionality of input coordinates");
            if (xDim == 0U) throw npstat::NpstatInvalidArgument(
                "In npstat::interpolateHistoND: can not interpolate "
                "zero-dimensional histograms");
            if (!(interpolationDegree == 0U ||
                  interpolationDegree == 1U ||
                  interpolationDegree == 3U)) throw npstat::NpstatInvalidArgument(
                "In npstat::interpolateHistoND: "
                "unsupported interpolation degree");
            if (interpolationDegree == 3U && !histo.isUniformlyBinned())
                throw npstat::NpstatInvalidArgument(
                    "In npstat::interpolateHistoND: unsupported "
                    "interpolation degree for non-uniform binning");
        }
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double *x, const unsigned xDim,
                             const unsigned interpolationDegree)
    {
        Private::iHND_checkArgs(histo, xDim, interpolationDegree);
        assert(x);
        const Axis* ax = &histo.axes()[0];
        double coords[CHAR_BIT*sizeof(unsigned long)];
        for (unsigned i=0; i<xDim; ++i)
            coords[i] = ax[i].fltBinNumber(x[i], false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, xDim);
        case 3U:
            return bins.interpolate3(coords, xDim);
        default:
            return bins.closest(coords, xDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 1U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const double coords = histo.axis(0).fltBinNumber(x0, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(&coords, expDim);
        case 3U:
            return bins.interpolate3(&coords, expDim);
        default:
            return bins.closest(&coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 2U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 3U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const double x3,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 4U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        coords[3] = ax[3].fltBinNumber(x3, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const double x3,
                             const double x4,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 5U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        coords[3] = ax[3].fltBinNumber(x3, false);
        coords[4] = ax[4].fltBinNumber(x4, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const double x3,
                             const double x4,
                             const double x5,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 6U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        coords[3] = ax[3].fltBinNumber(x3, false);
        coords[4] = ax[4].fltBinNumber(x4, false);
        coords[5] = ax[5].fltBinNumber(x5, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const double x3,
                             const double x4,
                             const double x5,
                             const double x6,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 7U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        coords[3] = ax[3].fltBinNumber(x3, false);
        coords[4] = ax[4].fltBinNumber(x4, false);
        coords[5] = ax[5].fltBinNumber(x5, false);
        coords[6] = ax[6].fltBinNumber(x6, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const double x3,
                             const double x4,
                             const double x5,
                             const double x6,
                             const double x7,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 8U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        coords[3] = ax[3].fltBinNumber(x3, false);
        coords[4] = ax[4].fltBinNumber(x4, false);
        coords[5] = ax[5].fltBinNumber(x5, false);
        coords[6] = ax[6].fltBinNumber(x6, false);
        coords[7] = ax[7].fltBinNumber(x7, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const double x3,
                             const double x4,
                             const double x5,
                             const double x6,
                             const double x7,
                             const double x8,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 9U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        coords[3] = ax[3].fltBinNumber(x3, false);
        coords[4] = ax[4].fltBinNumber(x4, false);
        coords[5] = ax[5].fltBinNumber(x5, false);
        coords[6] = ax[6].fltBinNumber(x6, false);
        coords[7] = ax[7].fltBinNumber(x7, false);
        coords[8] = ax[8].fltBinNumber(x8, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }

    template <typename Float, class Axis>
    Float interpolateHistoND(const HistoND<Float,Axis>& histo, 
                             const double x0,
                             const double x1,
                             const double x2,
                             const double x3,
                             const double x4,
                             const double x5,
                             const double x6,
                             const double x7,
                             const double x8,
                             const double x9,
                             const unsigned interpolationDegree)
    {
        const unsigned expDim = 10U;
        Private::iHND_checkArgs(histo, expDim, interpolationDegree);
        const Axis* ax = &histo.axes()[0];
        double coords[expDim];
        coords[0] = ax[0].fltBinNumber(x0, false);
        coords[1] = ax[1].fltBinNumber(x1, false);
        coords[2] = ax[2].fltBinNumber(x2, false);
        coords[3] = ax[3].fltBinNumber(x3, false);
        coords[4] = ax[4].fltBinNumber(x4, false);
        coords[5] = ax[5].fltBinNumber(x5, false);
        coords[6] = ax[6].fltBinNumber(x6, false);
        coords[7] = ax[7].fltBinNumber(x7, false);
        coords[8] = ax[8].fltBinNumber(x8, false);
        coords[9] = ax[9].fltBinNumber(x9, false);
        const ArrayND<Float>& bins(histo.binContents());
        switch (interpolationDegree)
        {
        case 1U:
            return bins.interpolate1(coords, expDim);
        case 3U:
            return bins.interpolate3(coords, expDim);
        default:
            return bins.closest(coords, expDim);
        }        
    }
}


#endif // NPSTAT_INTERPOLATEHISTOND_HH_

