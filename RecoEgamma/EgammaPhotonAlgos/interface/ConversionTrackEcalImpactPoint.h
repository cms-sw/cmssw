#ifndef ConversionTrackEcalImpactPoint_H
#define ConversionTrackEcalImpactPoint_H

/** \class ConversionTrackEcalImpactPoint
 *
 *
 * \author N. Marinelli - Univ. of Notre Dame
 *
 * \version   
 *
 ************************************************************/

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaTrackReco/interface/TrackSuperClusterAssociation.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
//
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
//
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"



class ConversionTrackEcalImpactPoint {

public:

  ConversionTrackEcalImpactPoint(const MagneticField* field );


  ~ConversionTrackEcalImpactPoint();

  std::vector<math::XYZPointF> find( const std::vector<reco::TransientTrack>& tracks, 
				    const edm::Handle<edm::View<reco::CaloCluster> >&  bcHandle );

  std::vector<reco::CaloClusterPtr> matchingBC() const {return matchingBC_;}
  void setMagneticField ( const MagneticField* mf ) { theMF_=mf;}
 

 

 
private:
  
  const MagneticField* theMF_;

  TrajectoryStateOnSurface  stateAtECAL_;

  mutable PropagatorWithMaterial*    forwardPropagator_ ;
  PropagationDirection       dir_;
  std::vector<reco::CaloClusterPtr> matchingBC_;

  static const ReferenceCountingPointer<BoundCylinder>  theBarrel_;
  static const ReferenceCountingPointer<BoundDisk>      theNegativeEtaEndcap_;
  static const ReferenceCountingPointer<BoundDisk>      thePositiveEtaEndcap_;

  static const BoundCylinder& barrel()        { return *theBarrel_;}
  static const BoundDisk& negativeEtaEndcap() { return *theNegativeEtaEndcap_;}
  static const BoundDisk& positiveEtaEndcap() { return *thePositiveEtaEndcap_;}

};

#endif // ConversionTrackEcalImpactPoint_H


