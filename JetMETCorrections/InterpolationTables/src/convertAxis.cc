#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "JetMETCorrections/InterpolationTables/interface/convertAxis.h"

namespace npstat {
    HistoAxis convertToHistoAxis(const UniformAxis& gridAxis)
    {
        const unsigned nBins = gridAxis.nCoords();
        const double xmin = gridAxis.min();
        const double xmax = gridAxis.max();
        const double hbw = 0.5*(xmax - xmin)/(nBins - 1U);
        HistoAxis ax(nBins, xmin-hbw, xmax+hbw, gridAxis.label().c_str());
        return ax;
    }
 
    UniformAxis convertToGridAxis(const HistoAxis& histoAxis)
    {
        const unsigned nBins = histoAxis.nBins();
        const double xmin = histoAxis.binCenter(0);
        const double xmax = histoAxis.binCenter(nBins - 1);
        UniformAxis ax(nBins, xmin, xmax, histoAxis.label().c_str());
        return ax;
    }

    NUHistoAxis convertToHistoAxis(const GridAxis& gridAxis, double xMin)
    {
        const unsigned nCoords = gridAxis.nCoords();
        std::vector<double> binEdges;
        binEdges.reserve(nCoords + 1U);
        binEdges.push_back(xMin);
        for (unsigned i=0; i<nCoords; ++i)
        {
            const double x = gridAxis.coordinate(i);
            if (x <= xMin)
                throw npstat::NpstatInvalidArgument("In npstat::convertToHistoAxis: "
                                            "conversion is impossible");
            const double halfbin = x - xMin;
            xMin = x + halfbin;
            binEdges.push_back(xMin);
        }
        NUHistoAxis ax(binEdges, gridAxis.label().c_str());
        return ax;
    }

    GridAxis convertToGridAxis(const NUHistoAxis& histoAxis)
    {
        const unsigned nBins = histoAxis.nBins();
        std::vector<double> coords;
        coords.reserve(nBins);
        for (unsigned i=0; i<nBins; ++i)
            coords.push_back(histoAxis.binCenter(i));
        GridAxis ax(coords, histoAxis.label().c_str());
        return ax;
    }
}
