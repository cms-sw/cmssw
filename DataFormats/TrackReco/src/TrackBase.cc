#include "Rtypes.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
#include <algorithm>

using namespace reco;
//TODO jaldeaar MOVE BACK TO INLINE ON .h
std::string TrackBase::algoName() const
{
    // I'd like to do:
    // return TrackBase::algoName(algorithm_);
    // but I cannot define a const static function. Why???

    switch (algorithm_) {
    case undefAlgorithm:
        return "undefAlgorithm";
    case ctf:
        return "ctf";
    case rs:
        return "rs";
    case cosmics:
        return "cosmics";
    case beamhalo:
        return "beamhalo";
    case initialStep:
        return "initialStep";
    case lowPtTripletStep:
        return "lowPtTripletStep";
    case pixelPairStep:
        return "pixelPairStep";
    case detachedTripletStep:
        return "detachedTripletStep";
    case mixedTripletStep:
        return "mixedTripletStep";
    case pixelLessStep:
        return "pixelLessStep";
    case tobTecStep:
        return "tobTecStep";
    case jetCoreRegionalStep:
        return "jetCoreRegionalStep";
    case iter8://TODO jaldeaar REMOVE?
        return "iter8";
    case muonSeededStepInOut:
        return "muonSeededStepInOut";
    case muonSeededStepOutInt:
        return "muonSeededStepOutIn";
    case outInEcalSeededConv:
        return "outInEcalSeededConv";
    case inOutEcalSeededConv:
        return "inOutEcalSeededConv";
    case nuclInter:
        return "nuclInter";
    case standAloneMuon:
        return "standAloneMuon";
    case globalMuon:
        return "globalMuon";
    case cosmicStandAloneMuon:
        return "cosmicStandAloneMuon";
    case cosmicGlobalMuon:
        return "cosmicGlobalMuon";
    case iter1LargeD0:
        return "iter1LargeD0";
    case iter2LargeD0:
        return "iter2LargeD0";
    case iter3LargeD0:
        return "iter3LargeD0";
    case iter4LargeD0:
        return "iter4LargeD0";
    case iter5LargeD0:
        return "iter5LargeD0";
    case bTagGhostTracks:
        return "bTagGhostTracks";
    case gsf:
        return "gsf";
    }
    return "undefAlgorithm";
}

// To be kept in synch with the enumerator definitions in TrackBase.h file
//TODO jaldeaar rename back to algoName
std::string const TrackBase::algorithmNames[] = {
    "undefAlgorithm",
    "ctf",
    "rs",
    "cosmics",
    "inialStep",
    "lowPtTripletStep",
    "pixelPairStep",
    "detachedTripletStep",
    "mixedTripletStep",
    "pixelLessStep",
    "tobTecStep",
    "jetCoreRegionalStep",
    "iter8",
    "muonSeededStepInOut",
    "muonSeededStepOutIn",
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
    int index = std::find(algorithmNames, algorithmNames + size, name) - algorithmNames;
    if (index == size) {
        return undefAlgorithm; // better this or throw() ?
    }

    // cast
    return TrackAlgorithm(index);
}
