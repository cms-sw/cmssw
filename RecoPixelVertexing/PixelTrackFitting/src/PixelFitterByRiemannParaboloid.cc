#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByRiemannParaboloid.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackBuilder.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackErrorParam.h"

#include "CommonTools/Utils/interface/DynArray.h"

using namespace std;


PixelFitterByRiemannParaboloid::PixelFitterByRiemannParaboloid(const edm::EventSetup* es,
                                                               const MagneticField* field,
                                                               bool useErrors,
                                                               bool useMultipleScattering)
    : es_(es), field_(field),
    useErrors_(useErrors), useMultipleScattering_(useMultipleScattering) {}

std::unique_ptr<reco::Track> PixelFitterByRiemannParaboloid::run(
    const std::vector<const TrackingRecHit*>& hits, const TrackingRegion& region) const {

  using namespace Rfit;

  std::unique_ptr<reco::Track> ret;

  unsigned int nhits = hits.size();

  if (nhits < 2) return ret;

  declareDynArray(GlobalPoint, nhits, points);
  declareDynArray(GlobalError, nhits, errors);
  declareDynArray(bool, nhits, isBarrel);

  for (unsigned int i = 0; i != nhits; ++i) {
    auto const& recHit = hits[i];
    points[i] = GlobalPoint(recHit->globalPosition().basicVector() - region.origin().basicVector());
    errors[i] = recHit->globalPositionError();
    isBarrel[i] = recHit->detUnit()->type().isBarrel();
  }

  Matrix<double, 3, Dynamic, 0, 3, max_nop> riemannHits(3, nhits);

  Matrix<double, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop> riemannHits_cov =
      MatrixXd::Zero(3 * nhits, 3 * nhits);

  for (unsigned int i = 0; i < nhits; ++i) {
    riemannHits.col(i) << points[i].x(), points[i].y(), points[i].z();

    const auto& errorMatrix = errors[i].matrix4D();

    for (auto j = 0; j < 3; ++j) {
      for (auto l = 0; l < 3; ++l) {
        riemannHits_cov(i + j * nhits, i + l * nhits) = errorMatrix(j, l);
      }
    }
  }

  float bField = 1 / PixelRecoUtilities::fieldInInvGev(*es_);
  helix_fit fittedTrack = Rfit::Helix_fit(riemannHits, riemannHits_cov, bField, useErrors_);
  int iCharge = fittedTrack.q;

  // parameters are:
  // 0: phi
  // 1: tip
  // 2: curvature
  // 3: cottheta
  // 4: zip
  float valPhi = fittedTrack.par(0);

  float valTip = fittedTrack.par(1);

  float valCotTheta = fittedTrack.par(3);

  float valZip = fittedTrack.par(4);
  float valPt = fittedTrack.par(2);
  //
  //  PixelTrackErrorParam param(valEta, valPt);
  float errValPhi = std::sqrt(fittedTrack.cov(0, 0));
  float errValTip = std::sqrt(fittedTrack.cov(1, 1));

  float errValPt = std::sqrt(fittedTrack.cov(2, 2));

  float errValCotTheta = std::sqrt(fittedTrack.cov(3, 3));
  float errValZip = std::sqrt(fittedTrack.cov(4, 4));

  float chi2 = fittedTrack.chi2_line + fittedTrack.chi2_circle;

  PixelTrackBuilder builder;
  Measurement1D phi(valPhi, errValPhi);
  Measurement1D tip(valTip, errValTip);

  Measurement1D pt(valPt, errValPt);
  Measurement1D cotTheta(valCotTheta, errValCotTheta);
  Measurement1D zip(valZip, errValZip);

  ret.reset(
      builder.build(pt, phi, cotTheta, tip, zip, chi2, iCharge, hits, field_, region.origin()));
  return ret;
}
