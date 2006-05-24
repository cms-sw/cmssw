#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

const GeomDetUnit * MuonTransientTrackingRecHit::detUnit() const
{
  return dynamic_cast<const GeomDetUnit*>(det());
}

GlobalVector MuonTransientTrackingRecHit::globalDirection() const
{
  return  (det()->surface().toGlobal(localDirection()));
}

GlobalError MuonTransientTrackingRecHit::globalDirectionError() const
{
  return ErrorFrameTransformer().transform( localDirectionError(), (det()->surface()));
}

bool MuonTransientTrackingRecHit::isDT() const{
  return  (geographicalId().subdetId() == MuonSubdetId::DT);
}

bool MuonTransientTrackingRecHit::isCSC() const{
  return  (geographicalId().subdetId() == MuonSubdetId::CSC);
}
