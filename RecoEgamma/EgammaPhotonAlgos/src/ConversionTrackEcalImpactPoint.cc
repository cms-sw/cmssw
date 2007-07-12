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

    LogDebug("ConversionTrackEcalImpactPoint") << " CTOR  " <<  "\n";  
    forwardPropagator_ = new PropagatorWithMaterial ( dir_ = alongMomentum, 0.000511, theMF_ );

}

ConversionTrackEcalImpactPoint::~ConversionTrackEcalImpactPoint() {

    LogDebug("ConversionTrackEcalImpactPoint") << " DTOR " <<  "\n";  
    delete    forwardPropagator_ ; 
    
}

std::vector<math::XYZPoint> ConversionTrackEcalImpactPoint::find( const std::vector<reco::TransientTrack>&  tracks  )   {
  std::vector<math::XYZPoint> result;
  
  std::cout << "  ConversionTrackEcalImpactPoint::find input tracks size  " << tracks.size() <<  "\n";

    
  
  for (    std::vector<reco::TransientTrack>::const_iterator iTk=tracks.begin(); iTk!=tracks.end(); ++iTk) {

    math::XYZPoint ecalImpactPosition(0.,0.,0.);
    const TrajectoryStateOnSurface myTSOS=(*iTk).innermostMeasurementState();
    if ( !( myTSOS.isValid() ) ) continue; 

    std::cout << "  ConversionTrackEcalImpactPoint::find  myTSOS is valid " <<  myTSOS << "\n";

    

    stateAtECAL_= forwardPropagator_->propagate( myTSOS, barrel() );

    if (!stateAtECAL_.isValid() ) {
      std::cout << "  ConversionTrackEcalImpactPoint::find  Barrel stateAtECAL_ is not valid " << std::endl;
    } else {
      std::cout << "  ConversionTrackEcalImpactPoint::find  Barrel stateAtECAL_ eta  " <<  stateAtECAL_.globalPosition().eta()  << " phi " << stateAtECAL_.globalPosition().phi()      << " " << stateAtECAL_.globalPosition() <<  "\n";
      
    }    
    

    if (!stateAtECAL_.isValid() || ( stateAtECAL_.isValid() && fabs(stateAtECAL_.globalPosition().eta() ) >1.479 )  ) {
      //   if (!stateAtECAL_.isValid()  ) {
         
      if ( (*iTk).innermostMeasurementState().globalPosition().eta() > 0.) {
	stateAtECAL_= forwardPropagator_->propagate( myTSOS, positiveEtaEndcap());
        if (!stateAtECAL_.isValid()  ) {
	  std::cout << "  ConversionTrackEcalImpactPoint::find  stateAtECAL_ not valid in positive endcap  " <<  "\n";
	} else {
	  std::cout << "  ConversionTrackEcalImpactPoint::find  stateAtECAL_ eta  ppsitive endcap  " <<  stateAtECAL_.globalPosition().eta()  << "\n";
	}	

      } else {

	stateAtECAL_= forwardPropagator_->propagate( myTSOS, negativeEtaEndcap());
        if (!stateAtECAL_.isValid()  ) {
	  std::cout << "  ConversionTrackEcalImpactPoint::find  stateAtECAL_ not valid in negative endcap  " <<  "\n";
	} else {
	  std::cout << "  ConversionTrackEcalImpactPoint::find  stateAtECAL_ eta  negative endcap  " <<  stateAtECAL_.globalPosition().eta()  << "\n";
	}

      }
    }
    
    
    if ( stateAtECAL_.isValid() ) {
      ecalImpactPosition = stateAtECAL_.globalPosition();
      std::cout << "  ConversionTrackEcalImpactPoint::find  output stateAtECAL_  is valid  eta   " << ecalImpactPosition.eta()  << " phi " << ecalImpactPosition.phi() << " " << ecalImpactPosition << "\n";
    }
    
    result.push_back(ecalImpactPosition  );
    



  }

  std::cout << " ConversionTrackEcalImpactPoint::find result size " << result.size() << std::endl;

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
