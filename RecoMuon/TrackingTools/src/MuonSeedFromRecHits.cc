/**
 *  See header file for a description of this class.
 *
 *
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author porting  R. Bellan
 *
 */
#include "RecoMuon/TrackingTools/interface/MuonSeedFromRecHits.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "gsl/gsl_statistics.h"

using namespace std;

template <class T> T sqr(const T& t) {return t*t;}

MuonSeedFromRecHits::MuonSeedFromRecHits()
: theField(nullptr)
{
}


TrajectorySeed MuonSeedFromRecHits::createSeed(float ptmean,
					       float sptmean,
					       ConstMuonRecHitPointer last) const
{
  
  const std::string metname = "Muon|RecoMuon|MuonSeedFromRecHits";

  MuonPatternRecoDumper debug;

  // FIXME: put it into a parameter set!
  double theMinMomentum = 3.0;
  int charge=std::copysign(1,ptmean);

  // Minimal pt
  if ( fabs(ptmean) < theMinMomentum ) ptmean = theMinMomentum * charge ;

  AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5,0) ;

  // Fill the LocalTrajectoryParameters
  LocalPoint segPos=last->localPosition();
  GlobalVector mom=last->globalPosition()-GlobalPoint();
  GlobalVector polar(GlobalVector::Spherical(mom.theta(),
                                             last->globalDirection().phi(),
                                             1.));
  polar *=fabs(ptmean)/polar.perp();
  LocalVector segDirFromPos=last->det()->toLocal(polar);

  LocalTrajectoryParameters param(segPos,segDirFromPos, charge);

  // this perform H.T() * parErr * H, which is the projection of the 
  // the measurement error (rechit rf) to the state error (TSOS rf)
  // Legenda:
  // H => is the 4x5 projection matrix
  // parError the 4x4 parameter error matrix of the RecHit

  // LogTrace(metname) << "Projection matrix:\n" << last->projectionMatrix();
  // LogTrace(metname) << "Error matrix:\n" << last->parametersError();

  mat = last->parametersError().similarityT( last->projectionMatrix() );
  

  float p_err = sqr(sptmean/(ptmean*ptmean));
  mat[0][0]= p_err;
  

  LocalTrajectoryError error(asSMatrix<5>(mat));
  
  // Create the TrajectoryStateOnSurface
  TrajectoryStateOnSurface tsos(param, error, last->det()->surface(), theField);

  // The following LogTraces must be moved somewhere else (StandAloneTrajectoryBuilder)
  // Here the TSOS does not have the magnetic field set, so dumpTSOS causes a crash
  // (when LogTrace/LogDebug is activated)
  //LogTrace(metname) << "Trajectory State on Surface before the extrapolation"<<endl;
  //LogTrace(metname) << debug.dumpTSOS(tsos);
  
  // Take the DetLayer on which relies the rechit
  DetId id = last->geographicalId();
  // Segment layer
  //LogTrace(metname) << "The RecSegment relies on: "<<endl;
  //LogTrace(metname) << debug.dumpMuonId(id);
  //LogTrace(metname) << debug.dumpTSOS(tsos);

  // Transform it in a TrajectoryStateOnSurface
  
  
  PTrajectoryStateOnDet const & seedTSOS =
    trajectoryStateTransform::persistentState( tsos ,id.rawId());
  
  edm::OwnVector<TrackingRecHit> container;
  for (unsigned l=0; l<theRhits.size(); l++) {
      container.push_back( theRhits[l]->hit()->clone() );
  }

  TrajectorySeed theSeed(seedTSOS,container,alongMomentum);
   
  return theSeed;
}




