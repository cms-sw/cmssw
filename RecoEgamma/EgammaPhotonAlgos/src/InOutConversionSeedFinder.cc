#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionBarrelEstimator.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionForwardEstimator.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/FastHelix.h"

// Field
#include "MagneticField/Engine/interface/MagneticField.h"
//
#include "CLHEP/Matrix/Matrix.h"
// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
//
//
#include "FWCore/Framework/interface/EventSetup.h"
//
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Geometry/Point3D.h"

InOutConversionSeedFinder::InOutConversionSeedFinder(  const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker) : ConversionSeedFinder( field, theInputMeasurementTracker )  {
  std::cout << "  InOutConversionSeedFinder CTOR " << std::endl;      
  theLayerMeasurements_ =  new LayerMeasurements(theInputMeasurementTracker );

  the2ndHitdphi_ = 0.008; 
  the2ndHitdzConst_ = 5.;
  the2ndHitdznSigma_ = 2.;


}



InOutConversionSeedFinder::~InOutConversionSeedFinder() {
  std::cout << " InOutConversionSeedFinder DTOR " << std::endl;
  delete theLayerMeasurements_;
}



void InOutConversionSeedFinder::makeSeeds( const reco::BasicClusterCollection* allBC )  const  {

  std::cout << "  InOutConversionSeedFinder::makeSeeds() " << std::endl;
  theSeeds_.clear();
  std::cout << " Check Basic cluster collection size " << allBC->size() << std::endl;  
  theSCPosition_= GlobalPoint ( theSC_->x(), theSC_->y(), theSC_->z() );
  bcCollection_=allBC;


  findLayers();


  fillClusterSeeds();
  std::cout << "Built vector of seeds of size  " << theSeeds_.size() <<  std::endl ;
 
  

  
  
}


void InOutConversionSeedFinder::fillClusterSeeds() const {

  vector<Trajectory>::const_iterator outInTrackItr;
 
  std::cout << "  InOutConversionSeedFinder::fillClusterSeeds outInTracks_.size " << theOutInTracks_.size() << std::endl;
  //Start looking for seeds for both of the 2 best tracks from the inward tracking


  for(outInTrackItr = theOutInTracks_.begin(); outInTrackItr != theOutInTracks_.end();  ++outInTrackItr) {
    std::cout << " InOutConversionSeedFinder::fillClusterSeeds track hits " << (*outInTrackItr).foundHits() << std::endl;
    
    //Find the first valid hit of the track
    // Measurements are ordered according to the direction in which the trajectories were built
    vector<TrajectoryMeasurement> measurements = (*outInTrackItr).measurements();
    
    
    vector<TrajectoryMeasurement>::iterator measurementItr;
    vector<const DetLayer*> allLayers=layerList();
    std::cout << "  InOutConversionSeedFinder::fillClusterSeed allLayers.size " <<  allLayers.size() << std::endl;
    for(unsigned int i = 0; i < allLayers.size(); ++i) {
      std::cout <<  " allLayers " << allLayers[i] << std::endl; 
    }



    vector<const DetLayer*> myLayers;
    int len=measurements.size();
    myLayers.resize(len);
    
    
    vector<TrajectoryMeasurement*> myItr;
    TrajectoryMeasurement* myPointer=0;
    std::cout << "  InOutConversionSeedFinder::fillClusterSeeds measurements.size " << measurements.size() <<std::endl;
 
    int iMea= measurements.size();
    for(measurementItr = measurements.begin(); measurementItr != measurements.end();  ++measurementItr) {
      iMea--;
      std::cout << "  InOutConversionSeedFinder::fillClusterSeeds measurement on  layer  " << measurementItr->layer() <<   " " <<&(*measurementItr) <<  std::endl;
      if( (*measurementItr).recHit()->isValid()) {
	//        myLayers.push_back( measurementItr->layer() );
        myLayers[iMea]= measurementItr->layer(); 
        myItr.push_back( &(*measurementItr) );
        
       
      }
    }
    
    

    std::cout << " InOutConversionSeedFinder::fillClusterSeed myLayers.size " <<  myLayers.size() << std::endl;
    for( unsigned int i = 0; i < myLayers.size(); ++i) {
      std::cout <<  " myLayers " << myLayers[i] << " myItr " << myItr[i] << std::endl; 
    }


    if ( myItr.size()==0 )  std::cout << "HORRENDOUS ERROR!  No meas on track!" << std::endl;
    
    unsigned int ilayer;
    for(ilayer = 0; ilayer < allLayers.size(); ++ilayer) {
      std::cout <<  " allLayers in the search loop  " << allLayers[ilayer] <<  " " << myLayers[0] <<  std::endl; 
      if ( allLayers[ilayer] == myLayers[0]) {
	std::cout <<  " allLayers in the search loop   allLayers[ilayer] == myLayers[0])  " << allLayers[ilayer] <<  " " << myLayers[0] <<  std::endl; 
        myPointer=myItr[0];

	std::cout << "Layer " << ilayer << "  contains the first valid measurement " << std::endl; 	
	printLayer(ilayer);	

	if ( (myLayers[0])->location() == GeomDetEnumerators::barrel ) {
	  const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(myLayers[0]);
	  std::cout << " InOutConversionSeedFinder::fillClusterSeeds  **** firstHit found in Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  endl;
	} else {
	  const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(myLayers[0]);
	  std::cout << " InOutwardConversionSeedFinder::fillClusterSeeds  **** firstHit found in Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  std::endl;
	}
	
	
	break;

      } else if ( allLayers[ilayer] == myLayers[1] )  {
        myPointer=myItr[1];

	std::cout << "Layer " << ilayer << "  contains the first valid measurement " << std::endl; 	
	if ( (myLayers[1])->location() == GeomDetEnumerators::barrel ) {
	  const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(myLayers[1]);

	} else {
	  const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(myLayers[1]);
	  std::cout << " InOutwardConversionSeedFinder::fillClusterSeeds  ****  2ndHitfound on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  std::endl;
	}



	break;

      }
    }
    


    if(ilayer == allLayers.size()) {
      cout << "InOutConversionSeedFinder::fillClusterSeeds ERROR could not find layer on list" << endl;
      return;
    }
    
    PropagatorWithMaterial reversePropagator(oppositeToMomentum, 0.000511, theMF_);
    //thePropagatorWithMaterial_.setPropagationDirection(oppositeToMomentum);
    FreeTrajectoryState * fts = myPointer->updatedState().freeTrajectoryState();
    std::cout << " InOutConversionSeedFinder::fillClusterSeeds First FTS charge " << fts->charge() << std::endl;


    while (ilayer > 0) {
      
      std::cout << " InOutConversionSeedFinder::fillClusterSeeds looking for 2nd seed from layer " << ilayer << std::endl;
      
      if ( (allLayers[ilayer])->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(allLayers[ilayer]);
      std::cout <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  std::endl;     
      } else {
	const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(allLayers[ilayer]);
	std::cout <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() << std::endl;
      }
      
      
      const DetLayer * previousLayer = allLayers[ilayer];
      // Propagate to the previous layer
      // The present layer is actually included in the loop so that a partner can be searched for
      // Applying the propagator to the same layer does not do any harm. It simply does nothing
      
      //      const Propagator& newProp=  thePropagatorWithMaterial_;
      const Propagator& newProp=reversePropagator;
      TrajectoryStateOnSurface  stateAtPreviousLayer= newProp.propagate(*fts, previousLayer->surface() );
      
      if ( stateAtPreviousLayer.isValid() ) {
	std::cout << "InOutConversionSeedFinder::fillClusterSeeds  Propagate back to layer "  << ilayer << std::endl;
	//std::cout << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer " << stateAtPreviousLayer << std::endl;
	//std:: cout << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer.globalDirection " << stateAtPreviousLayer.globalDirection()  << std::endl;
	
      }
      
      if(!stateAtPreviousLayer.isValid()) {
	std::cout << "InOutConversionSeedFinder::fillClusterSeeds ERROR:could not propagate back to layer "  << ilayer << std::endl;
	//std::cout << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer " << stateAtPreviousLayer <<std:: endl;
      } else {
	//std::cout << "stateAtPreviousLayer position" << 
	//         stateAtPreviousLayer.globalPosition() << std::endl;
	
	
	startSeed(fts,  stateAtPreviousLayer, -1, ilayer ); 
	
	//}
      }      
      
      --ilayer;
      
    }
   


    
  }  // End loop over Out In tracks
  

  
}



  void InOutConversionSeedFinder::startSeed( FreeTrajectoryState * fts, const TrajectoryStateOnSurface & stateAtPreviousLayer, int charge, int ilayer  )  const {

          std::cout << " InOutConversionSeedFinder::startSeed " << std::endl;
	  // Get a list of basic clusters that are consistent with a track 
          // starting at the assumed conversion point with opp. charge to the 
          // inward track.  Loop over these basic clusters.
	  track2Charge_ = charge*fts->charge();
	  std::vector<const reco::BasicCluster*> bcVec;
          std::cout << " InOutConversionSeedFinder charge assumed for the in-out track  " << track2Charge_ <<  std::endl;

	  bcVec = getSecondBasicClusters(stateAtPreviousLayer.globalPosition(),track2Charge_);
	  
	  std::vector<const reco::BasicCluster*>::iterator bcItr;
	  std::cout << " InOutConversionSeedFinder::fillClusterSeeds bcVec.size " << bcVec.size() << std::endl;

	  // debug
	  for(bcItr = bcVec.begin(); bcItr != bcVec.end(); ++bcItr) {

	    theSecondBC_ = *bcItr;
	    // std::cout << " InOutConversionSeedFinder::fillClusterSeeds bc eta " << theSecondBC_->position().eta() << " phi " <<  theSecondBC_->position().phi() << " x = " << 130.*cos(theSecondBC_->position().phi() )  << " y= " << 130.*sin(theSecondBC_->position().phi() ) << std::endl;
	  }
	  //

	  for(bcItr = bcVec.begin(); bcItr != bcVec.end(); ++bcItr) {

	    theSecondBC_ = *bcItr;
	    GlobalPoint bcPos((theSecondBC_->position()).x(),
			      (theSecondBC_->position()).y(),
                              (theSecondBC_->position()).z());

	    //	    std::cout << " InOutConversionSeedFinder::fillClusterSeed bc position x " << bcPos.x() << " y " <<  bcPos.y() << " z  " <<  bcPos.z() << " eta " <<  bcPos.eta() << " phi " <<  bcPos.phi() << std::endl;
	    GlobalVector dir = stateAtPreviousLayer.globalDirection();
	    GlobalPoint back1mm = stateAtPreviousLayer.globalPosition();
	   

	    back1mm -= dir.unit()*0.1;
	    FastHelix helix(bcPos, stateAtPreviousLayer.globalPosition(), back1mm, theMF_);

            	  

	    findSeeds(stateAtPreviousLayer, helix.stateAtVertex().transverseCurvature(), ilayer);
	    

	  }



  }



std::vector<const reco::BasicCluster*> InOutConversionSeedFinder::getSecondBasicClusters(const GlobalPoint & conversionPosition, float charge) const {

  const float pi=3.141592654;
  const float twopi=2*pi;
  std::vector<const reco::BasicCluster*> result;

  std::cout << " InOutConversionSeedFinder::getSecondBasicClusters" << endl;

  Geom::Phi<float> theConvPhi(conversionPosition.phi() );
 
 

  int nBc=0;
  for( reco::BasicClusterCollection::const_iterator bcItr = bcCollection_->begin(); bcItr != bcCollection_->end(); bcItr++) {
    Geom::Phi<float> theBcPhi(bcItr->position().phi());
    std::cout << "InOutConversionSeedFinder::getSecondBasicClusters  Basic cluster phi " << theBcPhi << std::endl;
    // Require phi of cluster to be consistent with the conversion 
    // position and the track charge
    
   
    if (fabs(theBcPhi-theConvPhi ) < .5 &&
        ((charge<0 && theBcPhi-theConvPhi >-.1) || 
         (charge>0 && theBcPhi-theConvPhi <.1))){
      // std::cout << " InOutConversionSeedFinder::getSecondBasicClusters  Adding bc pointer " << &(*bcItr) << "  to vector:" << std::endl;
   
      result.push_back(&(*bcItr));
    }



    
  }



  return result;


}



void InOutConversionSeedFinder::findSeeds(const TrajectoryStateOnSurface & startingState,
					  float transverseCurvature, 
					  int startingLayer) const {
 

  vector<const DetLayer*> allLayers=layerList();
  std::cout << " InOutConversionSeedFinder::findSeeds startingLayer " << startingLayer << endl;


  // create error matrix
  AlgebraicSymMatrix m(5,1) ;
  m[0][0] = 0.1; m[1][1] = 0.0001 ; m[2][2] = 0.0001 ;
  m[3][3] = 0.0001 ; m[4][4] = 0.001;

  // Make an FTS consistent with the start point, start direction and curvature
  FreeTrajectoryState fts(GlobalTrajectoryParameters(startingState.globalPosition(), 
						     startingState.globalDirection(),
						     double(transverseCurvature), 0, theMF_),
			                             CurvilinearTrajectoryError(m));
  
  std::cout << "  InOutConversionSeedFinder::findSeeds Initial FTS charge " << fts.charge() << endl;
  // PropagatorWithMaterial forwardPropagator(alongMomentum)
  thePropagatorWithMaterial_.setPropagationDirection(alongMomentum);
  float dphi = 0.01;
  float zrange = 5.;
  for( unsigned int ilayer = startingLayer; ilayer <= startingLayer+1 && (ilayer < allLayers.size()-2); ++ilayer) {
    const DetLayer * layer = allLayers[ilayer];
    
    
    
    ///// debug
    if ( layer->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(layer);
    std::cout << " InOutConversionSeedFinder::findSeeds  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  std::endl;     
    } else {
      const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(layer);
      std::cout << " InOutConversionSeedFinder::findSeeds  ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() << std:: endl;
    }
    //// end debug


    MeasurementEstimator * newEstimator=0;
    if (layer->location() == GeomDetEnumerators::barrel ) {
      //      cout << " InOutConversionSeedFinder::findSeeds Barrel ilayer " << ilayer << endl;
      newEstimator = new ConversionBarrelEstimator(-dphi, dphi, -zrange, zrange);
    }
    else {
        std::cout << " InOutConversionSeedFinder::findSeeds Forward  ilayer " << ilayer << endl;
        newEstimator = new ConversionForwardEstimator(-dphi, dphi, 15.);
    }
    

    theFirstMeasurements_.clear();
    // Get measurements compatible with the FTS and Estimator
    TSOS tsos(fts, layer->surface() );

    std::cout << " InOutConversionSeedFinder::findSeed propagationDirection " << int(thePropagatorWithMaterial_.propagationDirection() ) << std::endl;               
    theFirstMeasurements_ = theLayerMeasurements_->measurements( *layer, tsos, thePropagatorWithMaterial_, *newEstimator);
    delete newEstimator;
    std::cout <<  "InOutConversionSeedFinder::findSeeds  Found " << theFirstMeasurements_.size() << " first hits" << std::endl;

    //Loop over compatible hits
    int mea=0;
    for(vector<TrajectoryMeasurement>::iterator tmItr = theFirstMeasurements_.begin(); tmItr !=theFirstMeasurements_.end();  ++tmItr) {
     
      mea++;

      if (tmItr->recHit()->isValid() ) {
	// Make a new helix as in fillClusterSeeds() but using the hit position

        std::cout << " InOutConversionSeedFinder::findSeeds 1st hist position " << tmItr->recHit()->globalPosition() << std::endl;
	GlobalPoint bcPos((theSecondBC_->position()).x(),(theSecondBC_->position()).y(),(theSecondBC_->position()).z());
	GlobalVector dir = startingState.globalDirection();
	GlobalPoint back1mm = tmItr->recHit()->globalPosition();
	back1mm -= dir.unit()*0.1;
	FastHelix helix(bcPos,  tmItr->recHit()->globalPosition(), back1mm, theMF_);

        track2InitialMomentum_= helix.stateAtVertex().momentum();
	std::cout << "InOutConversionSeedFinder::findSeeds Updated estimatedPt = " << helix.stateAtVertex().momentum().perp()  << std::endl;
        //     << ", bcet = " << theBc->Et() 
        //     << ", estimatedPt/bcet = " << estimatedPt/theBc->Et() << endl;

	// Make a new FTS
	FreeTrajectoryState newfts(GlobalTrajectoryParameters(
							      tmItr->recHit()->globalPosition(), startingState.globalDirection(),
							      helix.stateAtVertex().transverseCurvature(), 0, theMF_), 
				   CurvilinearTrajectoryError(m));

	std::cout <<  " InOutConversionSeedFinder::findSeeds  new FTS charge " << newfts.charge() << std::endl;

	/*
        // Soome diagnostic output
        // may be useful - comparission of the basic cluster position 
	// with the ecal impact position of the track
	TrajectoryStateOnSurface stateAtECAL
	  = forwardPropagator.propagate(newfts, ECALSurfaces::barrel());
	if (!stateAtECAL.isValid() || abs(stateAtECAL.globalPosition().eta())>1.479) {
	  if (startingState.globalDirection().eta() > 0.) {
	    stateAtECAL = forwardPropagator.propagate(newfts, 
				      ECALSurfaces::positiveEtaEndcap());
	  } else {
	    stateAtECAL = forwardPropagator.propagate(newfts, 
				      ECALSurfaces::negativeEtaEndcap());
	  }
	}
	GlobalPoint ecalImpactPosition = stateAtECAL.isValid() ? stateAtECAL.globalPosition() : GlobalPoint(0.,0.,0.);
	cout << "Projected fts positon at ECAL surface: " << 
	  ecalImpactPosition << " bc position: " << theBc->Position() << endl;
	*/

        
	//        completeSeed(*tmItr, newfts,  &thePropagatorWithMaterial_, ilayer+1);
	// completeSeed(*tmItr, newfts,  &thePropagatorWithMaterial, ilayer+2);


      }

    }


    
  }


  
}





void InOutConversionSeedFinder::completeSeed(const TrajectoryMeasurement & m1,
FreeTrajectoryState & fts, const Propagator* propagator, int ilayer) const {

  std::cout <<  " InOutConversionSeedFinder::completeSeed ilayer " << ilayer <<  std::endl;

}


void InOutConversionSeedFinder::createSeed(const TrajectoryMeasurement & m1,  const TrajectoryMeasurement & m2) const {

  std::cout << " InOutConversionSeedFinder::createSeed " << std::endl;

}
