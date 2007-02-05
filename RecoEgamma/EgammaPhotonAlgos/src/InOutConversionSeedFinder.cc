#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionBarrelEstimator.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionForwardEstimator.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/FastHelix.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//
//
#include "FWCore/Framework/interface/EventSetup.h"
//
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Geometry/Point3D.h"

InOutConversionSeedFinder::InOutConversionSeedFinder(  const MagneticField* field, const MeasurementTracker* theInputMeasurementTracker ) :  ConversionSeedFinder( field, theInputMeasurementTracker )  {

  
  
  LogDebug(" InOutConversionSeedFinder") << " InOutConversionSeedFinder CTOR " << "\n";      
  theLayerMeasurements_ =  new LayerMeasurements(theInputMeasurementTracker );

  the2ndHitdphi_ = 0.008; 
  the2ndHitdzConst_ = 5.;
  the2ndHitdznSigma_ = 2.;


}



InOutConversionSeedFinder::~InOutConversionSeedFinder() {
  LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder DTOR " << "\n";
  delete theLayerMeasurements_;
}



void InOutConversionSeedFinder::makeSeeds( const reco::BasicClusterCollection& allBC )  const  {

  LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::makeSeeds() " << "\n";
  theSeeds_.clear();
  LogDebug("InOutConversionSeedFinder") << " Check Basic cluster collection size " << allBC.size() << "\n";  
  theSCPosition_= GlobalPoint ( theSC_->x(), theSC_->y(), theSC_->z() );
  bcCollection_= allBC;


  findLayers();


  fillClusterSeeds();
  LogDebug("InOutConversionSeedFinder") << "Built vector of seeds of size  " << theSeeds_.size() <<  "\n" ;
 
  

  
  
}


void InOutConversionSeedFinder::fillClusterSeeds() const {

  std::vector<Trajectory>::const_iterator outInTrackItr;
 
  LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::fillClusterSeeds outInTracks_.size " << theOutInTracks_.size() << "\n";
  //Start looking for seeds for both of the 2 best tracks from the inward tracking


  for(outInTrackItr = theOutInTracks_.begin(); outInTrackItr != theOutInTracks_.end();  ++outInTrackItr) {
    LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds out in input track hits " << (*outInTrackItr).foundHits() << "\n";
    
    //Find the first valid hit of the track
    // Measurements are ordered according to the direction in which the trajectories were built
    std::vector<TrajectoryMeasurement> measurements = (*outInTrackItr).measurements();
    
    
    std::vector<TrajectoryMeasurement>::iterator measurementItr;
    std::vector<const DetLayer*> allLayers=layerList();
    LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::fillClusterSeed allLayers.size " <<  allLayers.size() << "\n";
    for(unsigned int i = 0; i < allLayers.size(); ++i) {
      LogDebug("InOutConversionSeedFinder") <<  " allLayers " << allLayers[i] << "\n"; 
      printLayer(i);
    }



    std::vector<const DetLayer*> myLayers;
    myLayers.clear();    
    
    std::vector<TrajectoryMeasurement*> myItr;
    TrajectoryMeasurement* myPointer=0;
    LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::fillClusterSeeds measurements.size " << measurements.size() <<"\n";
 
 
 
    for(measurementItr = measurements.begin(); measurementItr != measurements.end();  ++measurementItr) {
 
      if( (*measurementItr).recHit()->isValid()) {
	
	LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::fillClusterSeeds measurement on  layer  " << measurementItr->layer() <<   " " <<&(*measurementItr) <<  " postion " << measurementItr->recHit()->globalPosition() <<   " R " << sqrt( measurementItr->recHit()->globalPosition().x()*measurementItr->recHit()->globalPosition().x() + measurementItr->recHit()->globalPosition().y()*measurementItr->recHit()->globalPosition().y() ) << "\n";
	
	        
	myLayers.push_back( measurementItr->layer() ) ; 
	myItr.push_back( &(*measurementItr) );
        
       
      }
    }
    
    

    LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeed myLayers.size " <<  myLayers.size() << "\n";
    for( unsigned int i = 0; i < myLayers.size(); ++i) {
      LogDebug("InOutConversionSeedFinder") <<  " myLayers " << myLayers[i] << " myItr " << myItr[i] << "\n"; 
    }


    if ( myItr.size()==0 )  LogDebug(" InOutConversionSeedFinder") << "HORRENDOUS ERROR!  No meas on track!" << "\n";
    
    unsigned int ilayer;
    for(ilayer = 0; ilayer < allLayers.size(); ++ilayer) {
      LogDebug("InOutConversionSeedFinder") <<  " allLayers in the search loop  " << allLayers[ilayer] <<  " " << myLayers[0] <<  "\n"; 
      if ( allLayers[ilayer] == myLayers[0]) {

        myPointer=myItr[0];

	LogDebug("InOutConversionSeedFinder") <<  " allLayers in the search loop   allLayers[ilayer] == myLayers[0])  " << allLayers[ilayer] <<  " " << myLayers[0] <<  " myPointer " << myPointer << "\n"; 

	LogDebug("InOutConversionSeedFinder") << "Layer " << ilayer << "  contains the first valid measurement " << "\n"; 	
	printLayer(ilayer);	

	if ( (myLayers[0])->location() == GeomDetEnumerators::barrel ) {
	  const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(myLayers[0]);
	  LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds  **** firstHit found in Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<   "\n";
	} else {
	  const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(myLayers[0]);
	  LogDebug("InOutConversionSeedFinder") << " InOutwardConversionSeedFinder::fillClusterSeeds  **** firstHit found in Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  "\n";
	}
	
	
	break;

      } else if ( allLayers[ilayer] == myLayers[1] )  {
        myPointer=myItr[1];

	LogDebug("InOutConversionSeedFinder") <<  " allLayers in the search loop   allLayers[ilayer] == myLayers[1])  " << allLayers[ilayer] <<  " " << myLayers[1] <<  " myPointer " << myPointer << "\n"; 

	LogDebug("InOutConversionSeedFinder") << "Layer " << ilayer << "  contains the first valid measurement " << "\n"; 	
	if ( (myLayers[1])->location() == GeomDetEnumerators::barrel ) {
	  const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(myLayers[1]);
	  LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds  **** 2ndHit found in Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<   "\n"; 
	} else {
	  const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(myLayers[1]);
	  LogDebug("InOutConversionSeedFinder") << " InOutwardConversionSeedFinder::fillClusterSeeds  ****  2ndHitfound on forw layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  "\n";
	}



	break;

      }
    }
    


    if(ilayer == allLayers.size()) {
       LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::fillClusterSeeds ERROR could not find layer on list" <<  "\n"; 
      return;
    }
    
    PropagatorWithMaterial reversePropagator(oppositeToMomentum, 0.000511, theMF_);
    //thePropagatorWithMaterial_.setPropagationDirection(oppositeToMomentum);
    FreeTrajectoryState * fts = myPointer->updatedState().freeTrajectoryState();
    LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds First FTS charge " << fts->charge() << " Position " << fts->position() << " momentum " << fts->momentum() << " R " << sqrt(fts->position().x()*fts->position().x() + fts->position().y()* fts->position().y() ) <<  "\n";


    while (ilayer > 0) {
      
      LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds looking for 2nd seed from layer " << ilayer << "\n";
      
      if ( (allLayers[ilayer])->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(allLayers[ilayer]);
      LogDebug("InOutConversionSeedFinder") <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  "\n";     
      } else {
	const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(allLayers[ilayer]);
	LogDebug("InOutConversionSeedFinder") <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() << "\n";
      }
      
      
      const DetLayer * previousLayer = allLayers[ilayer];
      LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds previousLayer->surface() position before  " <<allLayers[ilayer] << " " <<  previousLayer->surface().position() << " layer location " << previousLayer->location() << "\n";   
      // Propagate to the previous layer
      // The present layer is actually included in the loop so that a partner can be searched for
      // Applying the propagator to the same layer does not do any harm. It simply does nothing
      
      //      const Propagator& newProp=  thePropagatorWithMaterial_;
      const Propagator& newProp=reversePropagator;
      LogDebug("InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds reversepropagator direction " << newProp.propagationDirection()  << "\n";
      TrajectoryStateOnSurface  stateAtPreviousLayer= newProp.propagate(*fts, previousLayer->surface() );
      LogDebug(" InOutConversionSeedFinder") << " InOutConversionSeedFinder::fillClusterSeeds previousLayer->surface() position after " << previousLayer->surface().position() << " layer location " << previousLayer->location() << "\n";

      
      if ( stateAtPreviousLayer.isValid() ) {
	LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::fillClusterSeeds  Propagate back to layer "  << ilayer << "\n";
	//LogDebug(" InOutConversionSeedFinder") << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer " << stateAtPreviousLayer << "\n";
	//std:: cout << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer.globalDirection " << stateAtPreviousLayer.globalDirection()  << "\n";
	
      }
      
      if(!stateAtPreviousLayer.isValid()) {
	LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::fillClusterSeeds ERROR:could not propagate back to layer "  << ilayer << "\n";
	//LogDebug(" InOutConversionSeedFinder") << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer " << stateAtPreviousLayer <<std:: endl;
      } else {
	//LogDebug(" InOutConversionSeedFinder") << "stateAtPreviousLayer position" << 
	//         stateAtPreviousLayer.globalPosition() << "\n";
	
	
	startSeed(fts,  stateAtPreviousLayer, -1, ilayer ); 
	
	//}
      }      
      
      --ilayer;
      
    }
   


    
  }  // End loop over Out In tracks
  

  
}



  void InOutConversionSeedFinder::startSeed( FreeTrajectoryState * fts, const TrajectoryStateOnSurface & stateAtPreviousLayer, int charge, int ilayer  )  const {

          LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::startSeed " << "\n";
	  // Get a list of basic clusters that are consistent with a track 
          // starting at the assumed conversion point with opp. charge to the 
          // inward track.  Loop over these basic clusters.
	  track2Charge_ = charge*fts->charge();
	  std::vector<const reco::BasicCluster*> bcVec;
          LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder charge assumed for the in-out track  " << track2Charge_ <<  "\n";

	  bcVec = getSecondBasicClusters(stateAtPreviousLayer.globalPosition(),track2Charge_);
	  
	  std::vector<const reco::BasicCluster*>::iterator bcItr;
	  LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::fillClusterSeeds bcVec.size " << bcVec.size() << "\n";

	  // debug
	  for(bcItr = bcVec.begin(); bcItr != bcVec.end(); ++bcItr) {

	    //	    theSecondBC_ = *bcItr;
	    // LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::fillClusterSeeds bc eta " << theSecondBC_->position().eta() << " phi " <<  theSecondBC_->position().phi() << " x = " << 130.*cos(theSecondBC_->position().phi() )  << " y= " << 130.*sin(theSecondBC_->position().phi() ) << "\n";
	  }
	  //

	  for(bcItr = bcVec.begin(); bcItr != bcVec.end(); ++bcItr) {

	    theSecondBC_ = **bcItr;
	    //theSecondBC_ = *bcItr;
	    GlobalPoint bcPos((theSecondBC_.position()).x(),
			      (theSecondBC_.position()).y(),
                              (theSecondBC_.position()).z());

	    //	    LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::fillClusterSeed bc position x " << bcPos.x() << " y " <<  bcPos.y() << " z  " <<  bcPos.z() << " eta " <<  bcPos.eta() << " phi " <<  bcPos.phi() << "\n";
	    GlobalVector dir = stateAtPreviousLayer.globalDirection();
	    GlobalPoint back1mm = stateAtPreviousLayer.globalPosition();
	   

	    back1mm -= dir.unit()*0.1;
	    FastHelix helix(bcPos, stateAtPreviousLayer.globalPosition(), back1mm, theMF_);

            	  

	    findSeeds(stateAtPreviousLayer, helix.stateAtVertex().transverseCurvature(), ilayer);
	    

	  }



  }



std::vector<const reco::BasicCluster*> InOutConversionSeedFinder::getSecondBasicClusters(const GlobalPoint & conversionPosition, float charge) const {


  std::vector<const reco::BasicCluster*> result;

  LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::getSecondBasicClusters" <<  "\n"; 

  Geom::Phi<float> theConvPhi(conversionPosition.phi() );
 
 


  for( reco::BasicClusterCollection::const_iterator bcItr = bcCollection_.begin(); bcItr != bcCollection_.end(); bcItr++) {
    Geom::Phi<float> theBcPhi(bcItr->position().phi());
    LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::getSecondBasicClusters  Basic cluster phi " << theBcPhi << "\n";
    // Require phi of cluster to be consistent with the conversion 
    // position and the track charge
    
   
    if (fabs(theBcPhi-theConvPhi ) < .5 &&
        ((charge<0 && theBcPhi-theConvPhi >-.1) || 
         (charge>0 && theBcPhi-theConvPhi <.1))){
      // LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::getSecondBasicClusters  Adding bc pointer " << &(*bcItr) << "  to vector:" << "\n";
   
      result.push_back(&(*bcItr));
    }



    
  }



  return result;


}



void InOutConversionSeedFinder::findSeeds(const TrajectoryStateOnSurface & startingState,
					  float transverseCurvature, 
					  unsigned int startingLayer) const {
 

  std::vector<const DetLayer*> allLayers=layerList();
  LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::findSeeds starting forward propagation from  startingLayer " << startingLayer <<  "\n"; 


  // create error matrix
  AlgebraicSymMatrix m(5,1) ;
  m[0][0] = 0.1; m[1][1] = 0.0001 ; m[2][2] = 0.0001 ;
  m[3][3] = 0.0001 ; m[4][4] = 0.001;

  // Make an FTS consistent with the start point, start direction and curvature
  FreeTrajectoryState fts(GlobalTrajectoryParameters(startingState.globalPosition(), 
						     startingState.globalDirection(),
						     double(transverseCurvature), 0, theMF_),
			                             CurvilinearTrajectoryError(m));
  
  LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::findSeeds Initial FTS parameters " << fts <<  "\n"; 
  LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::findSeeds Initial FTS charge " << fts.charge() <<  "\n"; 

  thePropagatorWithMaterial_.setPropagationDirection(alongMomentum);

  float dphi = 0.01;
  float zrange = 5.;
  for( unsigned int ilayer = startingLayer; ilayer <= startingLayer+1 && (ilayer < allLayers.size()-2); ++ilayer) {
    const DetLayer * layer = allLayers[ilayer];
    
    
    
    ///// debug
    if ( layer->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(layer);
    LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::findSeeds  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  "\n";     
    } else {
      const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(layer);
      LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::findSeeds  ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  "\n"; 
    }
    //// end debug


    MeasurementEstimator * newEstimator=0;
    if (layer->location() == GeomDetEnumerators::barrel ) {
      //      cout << "InOutConversionSeedFinder::findSeeds Barrel ilayer " << ilayer << endl;
      newEstimator = new ConversionBarrelEstimator(-dphi, dphi, -zrange, zrange);
    }
    else {
        LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::findSeeds Forward  ilayer " << ilayer <<  "\n"; 
        newEstimator = new ConversionForwardEstimator(-dphi, dphi, 15.);
    }
    

    theFirstMeasurements_.clear();
    // Get measurements compatible with the FTS and Estimator
    TSOS tsos(fts, layer->surface() );

    LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::findSeed propagationDirection " << int(thePropagatorWithMaterial_.propagationDirection() ) << "\n";               
    theFirstMeasurements_ = theLayerMeasurements_->measurements( *layer, tsos, thePropagatorWithMaterial_, *newEstimator);

    delete newEstimator;
    LogDebug("InOutConversionSeedFinder") <<  "InOutConversionSeedFinder::findSeeds  Found " << theFirstMeasurements_.size() << " first hits" << "\n";

    //Loop over compatible hits
    int mea=0;
    for(std::vector<TrajectoryMeasurement>::iterator tmItr = theFirstMeasurements_.begin(); tmItr !=theFirstMeasurements_.end();  ++tmItr) {
     
      mea++;

      if (tmItr->recHit()->isValid() ) {
	// Make a new helix as in fillClusterSeeds() but using the hit position

        LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::findSeeds 1st hit position " << tmItr->recHit()->globalPosition() << " R " << sqrt(tmItr->recHit()->globalPosition().x()*tmItr->recHit()->globalPosition().x() + tmItr->recHit()->globalPosition().y()*tmItr->recHit()->globalPosition().y()   ) <<"\n";
	GlobalPoint bcPos((theSecondBC_.position()).x(),(theSecondBC_.position()).y(),(theSecondBC_.position()).z());
	GlobalVector dir = startingState.globalDirection();
	GlobalPoint back1mm = tmItr->recHit()->globalPosition();
	back1mm -= dir.unit()*0.1;
	FastHelix helix(bcPos,  tmItr->recHit()->globalPosition(), back1mm, theMF_);

        track2InitialMomentum_= helix.stateAtVertex().momentum();
	LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::findSeeds Updated estimatedPt = " << helix.stateAtVertex().momentum().perp()  << "\n";
        //     << ", bcet = " << theBc->Et() 
        //     << ", estimatedPt/bcet = " << estimatedPt/theBc->Et() << endl;

	// Make a new FTS
	FreeTrajectoryState newfts(GlobalTrajectoryParameters(
							      tmItr->recHit()->globalPosition(), startingState.globalDirection(),
							      helix.stateAtVertex().transverseCurvature(), 0, theMF_), 
				   CurvilinearTrajectoryError(m));

	LogDebug("InOutConversionSeedFinder") <<  "InOutConversionSeedFinder::findSeeds  new FTS charge " << newfts.charge() << "\n";

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

        
	completeSeed(*tmItr, newfts,  &thePropagatorWithMaterial_, ilayer+1);
	completeSeed(*tmItr, newfts,  &thePropagatorWithMaterial_, ilayer+2);


      }

    }


    
  }


  
}





void InOutConversionSeedFinder::completeSeed(const TrajectoryMeasurement & m1,
FreeTrajectoryState & fts, const Propagator* propagator, int ilayer) const {

  LogDebug("InOutConversionSeedFinder") <<  "InOutConversionSeedFinder::completeSeed ilayer " << ilayer <<  "\n";
 // A seed is made from 2 Trajectory Measuremennts.  The 1st is the input
  // argument m1.  This routine looks for the 2nd measurement in layer ilayer
  // Begin by making a new much stricter MeasurementEstimator based on the
  // position errors of the 1st hit.
  printLayer(ilayer);

  MeasurementEstimator * newEstimator;
  std::vector<const DetLayer*> allLayers=layerList();
  const DetLayer * layer = allLayers[ilayer];

  if (layer->location() == GeomDetEnumerators::barrel ) {
    
    float dz = sqrt(the2ndHitdznSigma_*the2ndHitdznSigma_*m1.recHit()->globalPositionError().czz()
		    + the2ndHitdzConst_*the2ndHitdzConst_);
    newEstimator = new ConversionBarrelEstimator(-the2ndHitdphi_, the2ndHitdphi_, -dz, dz);
    
  }
  else {
    float m1dr = sqrt(m1.recHit()->localPositionError().yy());
    float dr = sqrt(the2ndHitdznSigma_*the2ndHitdznSigma_*m1dr*m1dr
		    + the2ndHitdzConst_*the2ndHitdznSigma_);
    
    newEstimator =  new ConversionForwardEstimator(-the2ndHitdphi_, the2ndHitdphi_, dr);
  }


  TSOS tsos(fts, layer->surface() );
  LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::completeSeed propagationDirection  " << int(propagator->propagationDirection() ) << "\n";               
  LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::completeSeed pointer to estimator " << newEstimator << "\n";
  std::vector<TrajectoryMeasurement> measurements = theLayerMeasurements_->measurements( *layer, tsos, *propagator, *newEstimator);
  LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::completeSeed Found " << measurements.size() << " second hits " <<  "\n"; 
  delete newEstimator;
  
  for(unsigned int i = 0; i < measurements.size(); ++i) {
    if( measurements[i].recHit()->isValid()  ) {
      createSeed(m1, measurements[i]);
    }
  }






}



void InOutConversionSeedFinder::createSeed(const TrajectoryMeasurement & m1,  const TrajectoryMeasurement & m2) const {

  LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::createSeed " << "\n";

  GlobalTrajectoryParameters newgtp(  m1.recHit()->globalPosition(), track2InitialMomentum_, track2Charge_, theMF_ );
  CurvilinearTrajectoryError errors = m1.predictedState().curvilinearError();
  FreeTrajectoryState fts(newgtp, errors);
  TrajectoryStateOnSurface state1 = thePropagatorWithMaterial_.propagate(fts,  m1.recHit()->det()->surface());

  /*
  LogDebug("InOutConversionSeedFinder") << "hit surface " <<  m1.recHit()->det()->surface().position() << "\n";
  LogDebug("InOutConversionSeedFinder") << "prop to " << typeid( m1.recHit()->det()->surface() ).name() <<"\n";
  LogDebug("InOutConversionSeedFinder") << "prop to first hit " << state1 << "\n"; 
  LogDebug("InOutConversionSeedFinder") << "update to " <<  m1.recHit()->globalPosition() << "\n";
  */




  if ( state1.isValid() ) {

    TrajectoryStateOnSurface updatedState1 = theUpdator_.update(state1,  *m1.recHit() );

    if ( updatedState1.isValid() ) {

      TrajectoryStateOnSurface state2 = thePropagatorWithMaterial_.propagate(*updatedState1.freeTrajectoryState(),  m2.recHit()->det()->surface());
      
      if ( state2.isValid() ) {
	
	TrajectoryStateOnSurface updatedState2 = theUpdator_.update(state2, *m2.recHit() );
	TrajectoryMeasurement meas1(state1, updatedState1,  m1.recHit()  , m1.estimate(), m1.layer());
	TrajectoryMeasurement meas2(state2, updatedState2,  m2.recHit()  , m2.estimate(), m2.layer());
	
	edm::OwnVector<TrackingRecHit> myHits;
	myHits.push_back(meas1.recHit()->hit()->clone());
	myHits.push_back(meas2.recHit()->hit()->clone());
	
	LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::createSeed new seed " << "\n";
	
	TrajectoryStateTransform tsTransform;
	PTrajectoryStateOnDet* ptsod= tsTransform.persistentState(state2, meas2.recHit()->hit()->geographicalId().rawId()  );
	LogDebug("InOutConversionSeedFinder") << "  InOutConversionSeedFinder::createSeed New seed parameters " << state2 << "\n";
	theSeeds_.push_back(TrajectorySeed( *ptsod, myHits, alongMomentum )); 
	LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::createSeed New seed hit 1 position " << m1.recHit()->globalPosition() << "\n";
	LogDebug("InOutConversionSeedFinder") << "InOutConversionSeedFinder::createSeed New seed hit 2 position " << m2.recHit()->globalPosition() << "\n";


	
	
      }
    }
  }
    
}
