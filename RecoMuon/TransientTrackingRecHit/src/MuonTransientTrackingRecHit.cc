#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"


LocalVector MuonTransientTrackingRecHit::localDirection() const
{
  if(dynamic_cast<const RecSegment*>(hit()))
     return dynamic_cast<const RecSegment*>(hit())->localDirection(); 
  else return LocalVector(0.,0.,0.);

}

LocalError MuonTransientTrackingRecHit::localDirectionError() const
{
  if(dynamic_cast<const RecSegment*>(hit()))
     return dynamic_cast<const RecSegment*>(hit())->localDirectionError();
  else return LocalError(0.,0.,0.);

}

GlobalVector MuonTransientTrackingRecHit::globalDirection() const
{
  return  (det()->surface().toGlobal(localDirection()));
}

GlobalError MuonTransientTrackingRecHit::globalDirectionError() const
{
  return ErrorFrameTransformer().transform( localDirectionError(), (det()->surface()));
}

double MuonTransientTrackingRecHit::chi2() const 
{
  if(dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->chi2();
  else return 0.;
}

int MuonTransientTrackingRecHit::degreesOfFreedom() const 
{
  if(dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->degreesOfFreedom();
  else return 0;
}

bool MuonTransientTrackingRecHit::isDT() const{
  return  (geographicalId().subdetId() == MuonSubdetId::DT);
}

bool MuonTransientTrackingRecHit::isCSC() const{
  return  (geographicalId().subdetId() == MuonSubdetId::CSC);
}
