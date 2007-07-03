/** \file
 *
 *  $Date: 2007/03/07 16:18:45 $
 *  $Revision: 1.12 $
 */

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <map>

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::RecHitContainer   MuonRecHitContainer;


MuonTransientTrackingRecHit::MuonTransientTrackingRecHit(const GeomDet* geom, const TrackingRecHit* rh) :
  GenericTransientTrackingRecHit(geom,*rh){}

MuonTransientTrackingRecHit::MuonTransientTrackingRecHit(const MuonTransientTrackingRecHit& other ) :
  GenericTransientTrackingRecHit(other.det(), *(other.hit())) {}


LocalVector MuonTransientTrackingRecHit::localDirection() const {

  if (dynamic_cast<const RecSegment*>(hit()) )
     return dynamic_cast<const RecSegment*>(hit())->localDirection(); 
  else return LocalVector(0.,0.,0.);

}

LocalError MuonTransientTrackingRecHit::localDirectionError() const {

  if (dynamic_cast<const RecSegment*>(hit()))
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


AlgebraicSymMatrix MuonTransientTrackingRecHit::parametersError() const {
  
  AlgebraicSymMatrix err = GenericTransientTrackingRecHit::parametersError();
 
  if (det()->alignmentPositionError() != 0) {
    LocalError lape = ErrorFrameTransformer().transform(det()->alignmentPositionError()->globalError(), det()->surface());

    if(err.num_row() != 1 && err.num_row() != 2)
      throw cms::Exception("MuonTransientTrackingRecHit::parametersError") 
	<<"Wrong dimension of allignment error matrix: " << err.num_row() << std::endl;
    
    err[0][0] += lape.xx();
    
    if(err.num_row() == 2){
      err[0][1] += lape.xy();
      err[1][1] += lape.yy();
    }
  }  
  return err;
}

double MuonTransientTrackingRecHit::chi2() const 
{
  if (dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->chi2();
  else return 0.;
}

int MuonTransientTrackingRecHit::degreesOfFreedom() const 
{
  if (dynamic_cast<const RecSegment*>(hit()))
    return dynamic_cast<const RecSegment*>(hit())->degreesOfFreedom();
  else return 0;
}

bool MuonTransientTrackingRecHit::isDT() const{
  return  (geographicalId().subdetId() == MuonSubdetId::DT);
}

bool MuonTransientTrackingRecHit::isCSC() const{
  return  (geographicalId().subdetId() == MuonSubdetId::CSC);
}

bool MuonTransientTrackingRecHit::isRPC() const{
  return  (geographicalId().subdetId() == MuonSubdetId::RPC);
}

// FIXME, now it is "on-demand". I have to change it.
// FIXME check on mono hit!
TransientTrackingRecHit::ConstRecHitContainer MuonTransientTrackingRecHit::transientHits() const{

  ConstRecHitContainer theSubTransientRecHits;
  
  // the sub rec hit of this TransientRecHit
  std::vector<const TrackingRecHit*> ownRecHits = recHits();

  if(ownRecHits.size() == 0){
    theSubTransientRecHits.push_back(this);
    return theSubTransientRecHits;
  }
  
  // the components of the geom det on which reside this rechit
  std::vector<const GeomDet *> geomDets = det()->components();

  if(isDT() && dimension() == 2 && ownRecHits.front()->dimension() == 1 
     && (geomDets.size() == 3 || geomDets.size() == 2) ){ // it is a phi segment!!
    
    std::vector<const GeomDet *> subGeomDets;

    int sl = 1;
    for(std::vector<const GeomDet *>::const_iterator geoDet = geomDets.begin();
	geoDet != geomDets.end(); ++geoDet){
      if(sl != 3){ // FIXME!! this maybe is not always true
	std::vector<const GeomDet *> tmp = (*geoDet)->components();
	std::copy(tmp.begin(),tmp.end(),back_inserter(subGeomDets));
      }
      ++sl;
    }
    geomDets.clear();
    geomDets = subGeomDets;
  }
  
  // Fill the GeomDet map
  std::map<DetId,const GeomDet*> gemDetMap;
  
  for (std::vector<const GeomDet*>::const_iterator subDet = geomDets.begin(); 
       subDet != geomDets.end(); ++subDet)
    gemDetMap[ (*subDet)->geographicalId() ] = *subDet;
  
  std::map<DetId,const GeomDet*>::iterator gemDetMap_iter;
  
  // Loop in order to check the ids
  for (std::vector<const TrackingRecHit*>::const_iterator rechit = ownRecHits.begin(); 
       rechit != ownRecHits.end(); ++rechit){
    
    gemDetMap_iter = gemDetMap.find( (*rechit)->geographicalId() );
    
    if(gemDetMap_iter != gemDetMap.end() )
      theSubTransientRecHits.push_back(new MuonTransientTrackingRecHit(gemDetMap_iter->second, 
									 *rechit) );
    else if( (*rechit)->geographicalId() == det()->geographicalId() ) // Phi in DT is on Chamber
      theSubTransientRecHits.push_back(new MuonTransientTrackingRecHit(det(), 
								       *rechit) );
  }
  return theSubTransientRecHits;

}
