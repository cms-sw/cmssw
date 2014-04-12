#include <cmath>

#include "RecoJets/FFTJetAlgorithms/interface/EtaAndPtDependentPeakSelector.h"

namespace fftjetcms {
    EtaAndPtDependentPeakSelector::EtaAndPtDependentPeakSelector(
        std::istream& in) : ip_(fftjet::LinearInterpolator2d::read(in))
    {
    }

    EtaAndPtDependentPeakSelector::~EtaAndPtDependentPeakSelector()
    {
        delete ip_;
    }

    bool EtaAndPtDependentPeakSelector::operator()(const fftjet::Peak& peak)
        const
    {
        const double lookup = (*ip_)(peak.eta(), log(peak.scale()));
        return peak.magnitude() > exp(lookup);
    }

    EtaAndPtLookupPeakSelector::EtaAndPtLookupPeakSelector(
        unsigned nx, double xmin, double xmax,
        unsigned ny, double ymin, double ymax,
        const std::vector<double>& data)
        : lookupTable_(nx, xmin, xmax, ny, ymin, ymax, data)
    {
    }

    bool EtaAndPtLookupPeakSelector::operator()(const fftjet::Peak& peak)
        const
    {
        const double lookup = lookupTable_.closest(peak.eta(),
                                                   log(peak.scale()));
        return peak.magnitude() > exp(lookup);
    }
}
