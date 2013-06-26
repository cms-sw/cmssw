// $Id: FFTJetProducerSummary.cc,v 1.1 2010/11/22 23:29:11 igv Exp $

#include <algorithm>

#include "DataFormats/JetReco/interface/FFTJetProducerSummary.h"

namespace reco {
    FFTJetProducerSummary::FFTJetProducerSummary(
        const std::vector<double>& thresholds,
        const std::vector<unsigned>& levelOccupancy,
        const math::XYZTLorentzVector& unclustered,
        const std::vector<CandidatePtr>& constituents,
        double unused, double minScale, double maxScale,
        double scaleUsed, unsigned preclustersFound,
        unsigned iterationsPerformed, bool converged)
        : levelOccupancy_(levelOccupancy),
          unclustered_(unclustered),
          unclusConstituents_(constituents),
          unused_(unused),
          minScale_(minScale),
          maxScale_(maxScale),
          scaleUsed_(scaleUsed),
          preclustersFound_(preclustersFound),
          iterationsPerformed_(iterationsPerformed),
          converged_(converged)
    {
        thresholds_.resize(thresholds.size());
        std::copy(thresholds.begin(), thresholds.end(), thresholds_.begin());
    }
}
