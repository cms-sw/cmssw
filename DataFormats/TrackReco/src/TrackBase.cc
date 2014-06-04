#include "Rtypes.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
#include <algorithm>

using namespace reco;

//  Access the hit pattern, indicating in which Tracker layers the track has hits.
const HitPattern & TrackBase::hitPattern() const
{
    return hitPattern_;
}

bool TrackBase::appendHitPattern(const DetId &id, TrackingRecHit::Type hitType)
{
    return hitPattern_.appendHit(id, hitType);
}

bool TrackBase::appendHitPattern(const TrackingRecHit &hit)
{
    return hitPattern_.appendHit(hit);
}

// To be kept in synch with the enumerator definitions in TrackBase.h file
std::string const TrackBase::algoNames[] = {
    "undefAlgorithm",
    "ctf",
    "rs",
    "cosmics",
    "iter0",
    "iter1",
    "iter2",
    "iter3",
    "iter4",
    "iter5",
    "iter6",
    "iter7",
    "iter8",
    "iter9",
    "iter10",
    "outInEcalSeededConv",
    "inOutEcalSeededConv",
    "nuclInter",
    "standAloneMuon",
    "globalMuon",
    "cosmicStandAloneMuon",
    "cosmicGlobalMuon",
    "iter1LargeD0",
    "iter2LargeD0",
    "iter3LargeD0",
    "iter4LargeD0",
    "iter5LargeD0",
    "bTagGhostTracks",
    "beamhalo" ,
    "gsf"
};

std::string const TrackBase::qualityNames[] = {
    "loose",
    "tight",
    "highPurity",
    "confirmed",
    "goodIterative",
    "looseSetWithPV",
    "highPuritySetWithPV"
};

TrackBase::TrackBase() :
    chi2_(0),
    vertex_(0, 0, 0),
    momentum_(0, 0, 0),
    ndof_(0),
    charge_(0),
    algorithm_(undefAlgorithm),
    quality_(0),
    nLoops_(0)
{
    index idx = 0;
    for (index i = 0; i < dimension; ++i) {
        for (index j = 0; j <= i; ++j) {
            covariance_[idx++] = 0;
        }
    }
}

TrackBase::TrackBase(double chi2, double ndof, const Point &vertex, const Vector &momentum,
                     int charge, const CovarianceMatrix &cov, TrackAlgorithm algorithm,
                     TrackQuality quality, signed char nloops):
    chi2_(chi2),
    vertex_(vertex),
    momentum_(momentum),
    ndof_(ndof),
    charge_(charge),
    algorithm_(algorithm),
    quality_(0),
    nLoops_(nloops)
{
    index idx = 0;
    for (index i = 0; i < dimension; ++i) {
        for (index j = 0; j <= i; ++j) {
            covariance_[idx++] = cov(i, j);
        }
    }
    setQuality(quality);
}

TrackBase::~TrackBase()
{
    ;
}


TrackBase::CovarianceMatrix & TrackBase::fill(CovarianceMatrix &v) const
{
    return fillCovariance(v, covariance_);
}

TrackBase::TrackQuality TrackBase::qualityByName(const std::string &name)
{
    TrackQuality size = qualitySize;
    int index = std::find(qualityNames, qualityNames + size, name) - qualityNames;
    if (index == size) {
        return undefQuality; // better this or throw() ?
    }

    // cast
    return TrackQuality(index);
}

TrackBase::TrackAlgorithm TrackBase::algoByName(const std::string &name)
{
    TrackAlgorithm size = algoSize;
    int index = std::find(algoNames, algoNames + size, name) - algoNames;
    if (index == size) {
        return undefAlgorithm; // better this or throw() ?
    }

    // cast
    return TrackAlgorithm(index);
}
