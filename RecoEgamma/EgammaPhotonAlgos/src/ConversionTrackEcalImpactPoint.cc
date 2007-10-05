#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackEcalImpactPoint.h"
//#include "RecoEgamma/EgammaPhotonAlgos/interface/ECALSurfaces.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"


//
#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <vector>
#include <map>

ReferenceCountingPointer<BoundCylinder>  ConversionTrackEcalImpactPoint::theBarrel_ = 0;
ReferenceCountingPointer<BoundDisk>      ConversionTrackEcalImpactPoint::theNegativeEtaEndcap_ = 0;
ReferenceCountingPointer<BoundDisk>      ConversionTrackEcalImpactPoint::thePositiveEtaEndcap_ = 0;
bool                                     ConversionTrackEcalImpactPoint::theInit_ = false;


ConversionTrackEcalImpactPoint::ConversionTrackEcalImpactPoint(const MagneticField* field ): 

theMF_(field)
{ 

    forwardPropagator_ = new PropagatorWithMaterial ( dir_ = alongMomentum, 0.000511, theMF_ );

}

ConversionTrackEcalImpactPoint::~ConversionTrackEcalImpactPoint() {

    delete    forwardPropagator_ ; 
    
}

std::vector<math::XYZPoint> ConversionTrackEcalImpactPoint::find( const std::vector<reco::TransientTrack>&  tracks  )   {
  std::vector<math::XYZPoint> result;
  
  for (    std::vector<reco::TransientTrack>::const_iterator iTk=tracks.begin(); iTk!=tracks.end(); ++iTk) {

    math::XYZPoint ecalImpactPosition(0.,0.,0.);
    const TrajectoryStateOnSurface myTSOS=(*iTk).innermostMeasurementState();
    if ( !( myTSOS.isValid() ) ) continue; 

    stateAtECAL_= forwardPropagator_->propagate( myTSOS, barrel() );
    

    if (!stateAtECAL_.isValid() || ( stateAtECAL_.isValid() && fabs(stateAtECAL_.globalPosition().eta() ) >1.479 )  ) {
      //   if (!stateAtECAL_.isValid()  ) {
         
      if ( (*iTk).innermostMeasurementState().globalPosition().eta() > 0.) {
	stateAtECAL_= forwardPropagator_->propagate( myTSOS, positiveEtaEndcap());

      } else {

	stateAtECAL_= forwardPropagator_->propagate( myTSOS, negativeEtaEndcap());

      }
    }


    if ( stateAtECAL_.isValid() ) ecalImpactPosition = stateAtECAL_.globalPosition();


    result.push_back(ecalImpactPosition  );

  }

  return result;

}




void ConversionTrackEcalImpactPoint::initialize() {

  const float epsilon = 0.001;
  Surface::RotationType rot; // unit rotation matrix


  theBarrel_ = new BoundCylinder( Surface::PositionType(0,0,0), rot, 
				 SimpleCylinderBounds( barrelRadius()-epsilon, 
				       		       barrelRadius()+epsilon, 
						       -barrelHalfLength(), 
						       barrelHalfLength()));
  theNegativeEtaEndcap_ = 
    new BoundDisk( Surface::PositionType( 0, 0, -endcapZ()), rot, 
		   SimpleDiskBounds( 0, endcapRadius(), -epsilon, epsilon));
  
  thePositiveEtaEndcap_ = 
    new BoundDisk( Surface::PositionType( 0, 0, endcapZ()), rot, 
		   SimpleDiskBounds( 0, endcapRadius(), -epsilon, epsilon));
  
  theInit_ = true;


}
