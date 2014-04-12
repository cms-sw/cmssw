#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackEcalImpactPoint.h"
// Framework
//


//
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

std::vector<math::XYZPointF> ConversionTrackEcalImpactPoint::find( const std::vector<reco::TransientTrack>&  tracks,  const edm::Handle<edm::View<reco::CaloCluster> >&  bcHandle )   {

  
  std::vector<math::XYZPointF> result;
  // 
  matchingBC_.clear();   

  std::vector<reco::CaloClusterPtr> matchingBC(2);


  // 



  int iTrk=0;
  for (    std::vector<reco::TransientTrack>::const_iterator iTk=tracks.begin(); iTk!=tracks.end(); ++iTk) {

    math::XYZPointF ecalImpactPosition(0.,0.,0.);
    const TrajectoryStateOnSurface myTSOS=(*iTk).innermostMeasurementState();
    if ( !( myTSOS.isValid() ) ) continue; 

    stateAtECAL_= forwardPropagator_->propagate( myTSOS, barrel() );
    

    if (!stateAtECAL_.isValid() || ( stateAtECAL_.isValid() && fabs(stateAtECAL_.globalPosition().eta() ) >1.479 )  ) {
    
         
      if ( (*iTk).innermostMeasurementState().globalPosition().eta() > 0.) {
	stateAtECAL_= forwardPropagator_->propagate( myTSOS, positiveEtaEndcap());

      } else {

	stateAtECAL_= forwardPropagator_->propagate( myTSOS, negativeEtaEndcap());

      }
    }


    if ( stateAtECAL_.isValid() ) ecalImpactPosition = stateAtECAL_.globalPosition();


    result.push_back(ecalImpactPosition  );


    if ( stateAtECAL_.isValid() ) { 
      int goodBC=0;
      float bcDistanceToTrack=9999;
      reco::CaloClusterPtr matchedBCItr;
      int ibc=0;
      goodBC=0;

      for (unsigned i = 0; i < bcHandle->size(); ++i ) {
	float dEta= bcHandle->ptrAt(i)->position().eta() - ecalImpactPosition.eta()  ;
	float dPhi= bcHandle->ptrAt(i)->position().phi() - ecalImpactPosition.phi()  ;
	if ( sqrt(dEta*dEta + dPhi*dPhi)  <  bcDistanceToTrack ) {
          goodBC=ibc;
	  bcDistanceToTrack=sqrt(dEta*dEta + dPhi*dPhi);
	} 
        ibc++;	

      }

      matchingBC[iTrk]=bcHandle->ptrAt(goodBC);
    }
       
     
    iTrk++;
  }

  matchingBC_=matchingBC;

  return result;

}




void ConversionTrackEcalImpactPoint::initialize() {

  const float epsilon = 0.001;
  Surface::RotationType rot; // unit rotation matrix


  theBarrel_ = new Cylinder(barrelRadius(), Surface::PositionType(0,0,0), rot, 
				 new SimpleCylinderBounds( barrelRadius()-epsilon, 
				       		       barrelRadius()+epsilon, 
						       -barrelHalfLength(), 
						       barrelHalfLength()));
  theNegativeEtaEndcap_ = 
    new BoundDisk( Surface::PositionType( 0, 0, -endcapZ()), rot, 
		   new SimpleDiskBounds( 0, endcapRadius(), -epsilon, epsilon));
  
  thePositiveEtaEndcap_ = 
    new BoundDisk( Surface::PositionType( 0, 0, endcapZ()), rot, 
		   new SimpleDiskBounds( 0, endcapRadius(), -epsilon, epsilon));
  
  theInit_ = true;


}
