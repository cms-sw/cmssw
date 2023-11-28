#include "Rtypes.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
#include <algorithm>

using namespace reco;

// To be kept in synch with the enumerator definitions in TrackBase.h file
std::string const TrackBase::algoNames[] = {"undefAlgorithm",
                                            "ctf",
                                            "duplicateMerge",
                                            "cosmics",
                                            "initialStep",
                                            "lowPtTripletStep",
                                            "pixelPairStep",
                                            "detachedTripletStep",
                                            "mixedTripletStep",
                                            "pixelLessStep",
                                            "tobTecStep",
                                            "jetCoreRegionalStep",
                                            "conversionStep",
                                            "muonSeededStepInOut",
                                            "muonSeededStepOutIn",
                                            "outInEcalSeededConv",
                                            "inOutEcalSeededConv",
                                            "nuclInter",
                                            "standAloneMuon",
                                            "globalMuon",
                                            "cosmicStandAloneMuon",
                                            "cosmicGlobalMuon",
                                            "highPtTripletStep",
                                            "lowPtQuadStep",
                                            "detachedQuadStep",
                                            "displacedGeneralStep",
                                            "displacedRegionalStep",
                                            "bTagGhostTracks",
                                            "beamhalo",
                                            "gsf",
                                            "hltPixel",
                                            "hltIter0",
                                            "hltIter1",
                                            "hltIter2",
                                            "hltIter3",
                                            "hltIter4",
                                            "hltIterX",
                                            "hiRegitMuInitialStep",
                                            "hiRegitMuLowPtTripletStep",
                                            "hiRegitMuPixelPairStep",
                                            "hiRegitMuDetachedTripletStep",
                                            "hiRegitMuMixedTripletStep",
                                            "hiRegitMuPixelLessStep",
                                            "hiRegitMuTobTecStep",
                                            "hiRegitMuMuonSeededStepInOut",
                                            "hiRegitMuMuonSeededStepOutIn"};

std::string const TrackBase::qualityNames[] = {
    "loose", "tight", "highPurity", "confirmed", "goodIterative", "looseSetWithPV", "highPuritySetWithPV", "discarded"};

TrackBase::TrackBase()
    : covt0t0_(-1.f),
      covbetabeta_(-1.f),
      chi2_(0),
      vertex_(0, 0, 0),
      t0_(0),
      momentum_(0, 0, 0),
      beta_(0),
      ndof_(0),
      charge_(0),
      algorithm_(undefAlgorithm),
      originalAlgorithm_(undefAlgorithm),
      quality_(0),
      nLoops_(0),
      stopReason_(0) {
  algoMask_.set(algorithm_);
  index idx = 0;
  for (index i = 0; i < dimension; ++i) {
    for (index j = 0; j <= i; ++j) {
      covariance_[idx++] = 0;
    }
  }
}

TrackBase::TrackBase(double chi2,
                     double ndof,
                     const Point &vertex,
                     const Vector &momentum,
                     int charge,
                     const CovarianceMatrix &cov,
                     TrackAlgorithm algorithm,
                     TrackQuality quality,
                     signed char nloops,
                     uint8_t stopReason,
                     float t0,
                     float beta,
                     float covt0t0,
                     float covbetabeta)
    : covt0t0_(covt0t0),
      covbetabeta_(covbetabeta),
      chi2_(chi2),
      vertex_(vertex),
      t0_(t0),
      momentum_(momentum),
      beta_(beta),
      ndof_(ndof),
      charge_(charge),
      algorithm_(algorithm),
      originalAlgorithm_(algorithm),
      quality_(0),
      nLoops_(nloops),
      stopReason_(stopReason) {
  algoMask_.set(algorithm_);

  index idx = 0;
  for (index i = 0; i < dimension; ++i) {
    for (index j = 0; j <= i; ++j) {
      covariance_[idx++] = cov(i, j);
    }
  }
  setQuality(quality);
}

TrackBase::~TrackBase() { ; }

TrackBase::CovarianceMatrix &TrackBase::fill(CovarianceMatrix &v) const { return fillCovariance(v, covariance_); }

TrackBase::TrackQuality TrackBase::qualityByName(const std::string &name) {
  TrackQuality size = qualitySize;
  int index = std::find(qualityNames, qualityNames + size, name) - qualityNames;
  if (index == size) {
    return undefQuality;  // better this or throw() ?
  }

  // cast
  return TrackQuality(index);
}

TrackBase::TrackAlgorithm TrackBase::algoByName(const std::string &name) {
  TrackAlgorithm size = algoSize;
  int index = std::find(algoNames, algoNames + size, name) - algoNames;
  if (index == size) {
    return undefAlgorithm;  // better this or throw() ?
  }

  // cast
  return TrackAlgorithm(index);
}

double TrackBase::dxyError(Point const &vtx, math::Error<3>::type const &vertexCov) const {
  // Gradient of TrackBase::dxy(const Point &myBeamSpot) with respect to track parameters. Using unrolled expressions to avoid calling for higher dimension matrices
  // ( 0, 0, x_vert * cos(phi) + y_vert * sin(phi), 1, 0 )
  // Gradient with respect to point parameters
  // ( sin(phi), -cos(phi))
  // Propagate covariance assuming cross-terms of the covariance between track and vertex parameters are 0
  return std::sqrt((vtx.x() * px() + vtx.y() * py()) * (vtx.x() * px() + vtx.y() * py()) / (pt() * pt()) *
                       covariance(i_phi, i_phi) +
                   2 * (vtx.x() * px() + vtx.y() * py()) / pt() * covariance(i_phi, i_dxy) + covariance(i_dxy, i_dxy) +
                   py() * py() / (pt() * pt()) * vertexCov(0, 0) - 2 * py() * px() / (pt() * pt()) * vertexCov(0, 1) +
                   px() * px() / (pt() * pt()) * vertexCov(1, 1));
}
