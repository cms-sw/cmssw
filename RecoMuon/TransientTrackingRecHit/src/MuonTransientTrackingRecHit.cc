/** \file
 *
 */

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/AlignmentPositionError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <map>

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::RecHitContainer MuonRecHitContainer;

MuonTransientTrackingRecHit::MuonTransientTrackingRecHit(const GeomDet* geom, const TrackingRecHit* rh)
    : GenericTransientTrackingRecHit(*geom, *rh) {}

MuonTransientTrackingRecHit::MuonTransientTrackingRecHit(const MuonTransientTrackingRecHit& other)
    : GenericTransientTrackingRecHit(*other.det(), *(other.hit())) {}

LocalVector MuonTransientTrackingRecHit::localDirection() const {
  if (dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->localDirection();
  else
    return LocalVector(0., 0., 0.);
}

LocalError MuonTransientTrackingRecHit::localDirectionError() const {
  if (dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->localDirectionError();
  else
    return LocalError(0., 0., 0.);
}

GlobalVector MuonTransientTrackingRecHit::globalDirection() const {
  return (det()->surface().toGlobal(localDirection()));
}

GlobalError MuonTransientTrackingRecHit::globalDirectionError() const {
  return ErrorFrameTransformer().transform(localDirectionError(), (det()->surface()));
}

AlgebraicSymMatrix MuonTransientTrackingRecHit::parametersError() const {
  AlgebraicSymMatrix err = GenericTransientTrackingRecHit::parametersError();
  AlgebraicVector par = GenericTransientTrackingRecHit::parameters();

  const AlignmentPositionError* APE = det()->alignmentPositionError();
  if (APE != nullptr) {
    AlgebraicVector positions(2, 0);
    AlgebraicVector directions(2, 0);

    if (err.num_row() == 1) {
      positions[0] = 0.;
      positions[1] = 0.;
      directions[0] = 0.;
      directions[1] = 0.;
      LocalErrorExtended lape = ErrorFrameTransformer().transform46(APE->globalError(), positions, directions);
      err[0][0] += lape.cxx();
    } else if (err.num_row() == 2) {
      positions[0] = localPosition().x();
      positions[1] = 0.;
      directions[0] = 0.;
      directions[1] = 0.;
      LocalErrorExtended lape = ErrorFrameTransformer().transform46(APE->globalError(), positions, directions);

      AlgebraicSymMatrix lapeMatrix(2, 0);
      lapeMatrix[1][1] = lape.cxx();
      lapeMatrix[0][0] = lape.cphixphix();
      lapeMatrix[0][1] = lape.cphixx();

      if (err.num_row() != lapeMatrix.num_row())
        throw cms::Exception("MuonTransientTrackingRecHit::parametersError")
            << "Discrepancy between alignment error matrix and error matrix: APE " << lapeMatrix.num_row()
            << ", error matrix " << err.num_row() << std::endl;

      err += lapeMatrix;
    } else if (err.num_row() == 4) {
      positions[0] = par[2];
      positions[1] = par[3];
      directions[0] = par[0];
      directions[1] = par[1];

      LocalErrorExtended lape = ErrorFrameTransformer().transform46(APE->globalError(), positions, directions);

      AlgebraicSymMatrix lapeMatrix(4, 0);
      lapeMatrix[2][2] = lape.cxx();
      lapeMatrix[2][3] = lape.cyx();
      lapeMatrix[3][3] = lape.cyy();
      lapeMatrix[0][0] = lape.cphixphix();
      lapeMatrix[0][1] = lape.cphiyphix();
      lapeMatrix[1][1] = lape.cphiyphiy();

      lapeMatrix[0][2] = lape.cphixx();
      lapeMatrix[0][3] = lape.cphixy();
      lapeMatrix[1][3] = lape.cphiyy();
      lapeMatrix[1][2] = lape.cphiyx();

      if (err.num_row() != lapeMatrix.num_row())
        throw cms::Exception("MuonTransientTrackingRecHit::parametersError")
            << "Discrepancy between alignment error matrix and error matrix: APE " << lapeMatrix.num_row()
            << ", error matrix " << err.num_row() << std::endl;

      err += lapeMatrix;
    }
  }
  return err;
}

double MuonTransientTrackingRecHit::chi2() const {
  if (dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->chi2();
  else
    return 0.;
}

int MuonTransientTrackingRecHit::degreesOfFreedom() const {
  if (dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->degreesOfFreedom();
  else
    return 0;
}

bool MuonTransientTrackingRecHit::isDT() const { return (geographicalId().subdetId() == MuonSubdetId::DT); }

bool MuonTransientTrackingRecHit::isCSC() const { return (geographicalId().subdetId() == MuonSubdetId::CSC); }

bool MuonTransientTrackingRecHit::isGEM() const { return (geographicalId().subdetId() == MuonSubdetId::GEM); }

bool MuonTransientTrackingRecHit::isME0() const { return (geographicalId().subdetId() == MuonSubdetId::ME0); }

bool MuonTransientTrackingRecHit::isRPC() const { return (geographicalId().subdetId() == MuonSubdetId::RPC); }

// FIXME, now it is "on-demand". I have to change it.
// FIXME check on mono hit!
TransientTrackingRecHit::ConstRecHitContainer MuonTransientTrackingRecHit::transientHits() const {
  ConstRecHitContainer theSubTransientRecHits;

  // the sub rec hit of this TransientRecHit
  std::vector<const TrackingRecHit*> ownRecHits = recHits();

  if (ownRecHits.empty()) {
    theSubTransientRecHits.push_back(TransientTrackingRecHit::RecHitPointer(clone()));
    return theSubTransientRecHits;
  }

  // the components of the geom det on which reside this rechit
  std::vector<const GeomDet*> geomDets = det()->components();

  if (isDT() && dimension() == 2 && ownRecHits.front()->dimension() == 1 &&
      (geomDets.size() == 3 || geomDets.size() == 2)) {  // it is a phi segment!!

    std::vector<const GeomDet*> subGeomDets;

    int sl = 1;
    for (std::vector<const GeomDet*>::const_iterator geoDet = geomDets.begin(); geoDet != geomDets.end(); ++geoDet) {
      if (sl != 3) {  // FIXME!! this maybe is not always true
        std::vector<const GeomDet*> tmp = (*geoDet)->components();
        std::copy(tmp.begin(), tmp.end(), back_inserter(subGeomDets));
      }
      ++sl;
    }
    geomDets.clear();
    geomDets = subGeomDets;
  }

  // Fill the GeomDet map
  std::map<DetId, const GeomDet*> gemDetMap;

  for (std::vector<const GeomDet*>::const_iterator subDet = geomDets.begin(); subDet != geomDets.end(); ++subDet)
    gemDetMap[(*subDet)->geographicalId()] = *subDet;

  std::map<DetId, const GeomDet*>::iterator gemDetMap_iter;

  // Loop in order to check the ids
  for (std::vector<const TrackingRecHit*>::const_iterator rechit = ownRecHits.begin(); rechit != ownRecHits.end();
       ++rechit) {
    gemDetMap_iter = gemDetMap.find((*rechit)->geographicalId());

    if (gemDetMap_iter != gemDetMap.end())
      theSubTransientRecHits.push_back(
          TransientTrackingRecHit::RecHitPointer(new MuonTransientTrackingRecHit(gemDetMap_iter->second, *rechit)));
    else if ((*rechit)->geographicalId() == det()->geographicalId())  // Phi in DT is on Chamber
      theSubTransientRecHits.push_back(
          TransientTrackingRecHit::RecHitPointer(new MuonTransientTrackingRecHit(det(), *rechit)));
  }
  return theSubTransientRecHits;
}

void MuonTransientTrackingRecHit::invalidateHit() {
  setType(bad);
  trackingRecHit_->setType(bad);

  if (isDT()) {
    if (dimension() > 1) {                             // MB4s have 2D, but formatted in 4D segments
      std::vector<TrackingRecHit*> seg2D = recHits();  // 4D --> 2D
      // load 1D hits (2D --> 1D)
      for (std::vector<TrackingRecHit*>::iterator it = seg2D.begin(); it != seg2D.end(); ++it) {
        std::vector<TrackingRecHit*> hits1D = (*it)->recHits();
        (*it)->setType(bad);
        for (std::vector<TrackingRecHit*>::iterator it2 = hits1D.begin(); it2 != hits1D.end(); ++it2)
          (*it2)->setType(bad);
      }
    }
  } else if (isCSC())
    if (dimension() == 4) {
      std::vector<TrackingRecHit*> hits = recHits();  // load 2D hits (4D --> 1D)
      for (std::vector<TrackingRecHit*>::iterator it = hits.begin(); it != hits.end(); ++it)
        (*it)->setType(bad);
    }
}
