#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionBarrelEstimator.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionForwardEstimator.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionFastHelix.h"

// Field
//
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
// Geometry
//
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

//
//
//

InOutConversionSeedFinder::InOutConversionSeedFinder(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC)
    : ConversionSeedFinder(conf, iC), conf_(conf) {
  // std::cout << " InOutConversionSeedFinder CTOR " << "\n";

  maxNumberOfInOutSeedsPerInputTrack_ = conf_.getParameter<int>("maxNumOfSeedsInOut");
  //the2ndHitdphi_ = 0.008;
  the2ndHitdphi_ = 0.01;
  the2ndHitdzConst_ = 5.;
  the2ndHitdznSigma_ = 2.;
}

InOutConversionSeedFinder::~InOutConversionSeedFinder() {
  //std::cout << " InOutConversionSeedFinder DTOR " << "\n";
}

void InOutConversionSeedFinder::makeSeeds(const edm::Handle<edm::View<reco::CaloCluster> >& allBC) {
  //std::cout << "  InOutConversionSeedFinder::makeSeeds() " << "\n";
  theSeeds_.clear();
  //std::cout << " Check Calo cluster collection size " << allBC->size() << "\n";
  bcCollection_ = allBC;

  findLayers();

  fillClusterSeeds();
  //std::cout << "Built vector of seeds of size  " << theSeeds_.size() <<  "\n" ;

  theOutInTracks_.clear();
  inputTracks_.clear();
  theFirstMeasurements_.clear();
}

void InOutConversionSeedFinder::fillClusterSeeds() {
  std::vector<Trajectory>::const_iterator outInTrackItr;

  //std::cout << "  InOutConversionSeedFinder::fillClusterSeeds outInTracks_.size " << theOutInTracks_.size() << "\n";
  //Start looking for seeds for both of the 2 best tracks from the inward tracking

  ///// This bit is for debugging; it will go away
  /*
  for(outInTrackItr = theOutInTracks_.begin(); outInTrackItr != theOutInTracks_.end();  ++outInTrackItr) {


   //std::cout << " InOutConversionSeedFinder::fillClusterSeeds out in input track hits " << (*outInTrackItr).foundHits() << "\n";
    DetId tmpId = DetId( (*outInTrackItr).seed().startingState().detId());
    const GeomDet* tmpDet  = this->getMeasurementTracker()->geomTracker()->idToDet( tmpId );
    GlobalVector gv = tmpDet->surface().toGlobal( (*outInTrackItr).seed().startingState().parameters().momentum() );


   //std::cout << " InOutConversionSeedFinder::fillClusterSeed was built from seed position " <<gv   <<  " charge " << (*outInTrackItr).seed().startingState().parameters().charge() << "\n";

    Trajectory::DataContainer m=  outInTrackItr->measurements();
    int nHit=0;
    for (Trajectory::DataContainer::iterator itm = m.begin(); itm != m.end(); ++itm) {
      if ( itm->recHit()->isValid()  ) {
        nHit++;
	//std::cout << nHit << ")  Valid RecHit global position " << itm->recHit()->globalPosition() << " R " <<  itm->recHit()->globalPosition().perp() << " phi " << itm->recHit()->globalPosition().phi() << " eta " << itm->recHit()->globalPosition().eta() << "\n";
      } 

    }

  }

  */

  //Start looking for seeds for both of the 2 best tracks from the inward tracking
  for (outInTrackItr = theOutInTracks_.begin(); outInTrackItr != theOutInTracks_.end(); ++outInTrackItr) {
    //std::cout << " InOutConversionSeedFinder::fillClusterSeeds out in input track hits " << (*outInTrackItr).foundHits() << "\n";
    nSeedsPerInputTrack_ = 0;

    //Find the first valid hit of the track
    // Measurements are ordered according to the direction in which the trajectories were built
    std::vector<TrajectoryMeasurement> measurements = (*outInTrackItr).measurements();

    std::vector<const DetLayer*> allLayers = layerList();

    //std::cout << "  InOutConversionSeedFinder::fill clusterSeed allLayers.size " <<  allLayers.size() << "\n";
    for (unsigned int i = 0; i < allLayers.size(); ++i) {
      //std::cout <<  " allLayers " << allLayers[i] << "\n";
      printLayer(i);
    }

    std::vector<const DetLayer*> myLayers;
    myLayers.clear();
    std::vector<TrajectoryMeasurement>::reverse_iterator measurementItr;
    std::vector<TrajectoryMeasurement*> myItr;
    // TrajectoryMeasurement* myPointer=0;
    myPointer = nullptr;
    //std::cout << "  InOutConversionSeedFinder::fillClusterSeeds measurements.size " << measurements.size() <<"\n";

    for (measurementItr = measurements.rbegin(); measurementItr != measurements.rend(); ++measurementItr) {
      if ((*measurementItr).recHit()->isValid()) {
        //std::cout << "  InOutConversionSeedFinder::fillClusterSeeds measurement on  layer  " << measurementItr->layer() <<   " " <<&(*measurementItr) <<  " position " << measurementItr->recHit()->globalPosition() <<   " R " << sqrt( measurementItr->recHit()->globalPosition().x()*measurementItr->recHit()->globalPosition().x() + measurementItr->recHit()->globalPosition().y()*measurementItr->recHit()->globalPosition().y() ) << " Z " << measurementItr->recHit()->globalPosition().z() << " phi " <<  measurementItr->recHit()->globalPosition().phi() << "\n";

        myLayers.push_back(measurementItr->layer());
        myItr.push_back(&(*measurementItr));
      }
    }

    //std::cout << " InOutConversionSeedFinder::fillClusterSeed myLayers.size " <<  myLayers.size() << "\n";
    //    for( unsigned int i = 0; i < myLayers.size(); ++i) {
    ////std::cout <<  " myLayers " << myLayers[i] << " myItr " << myItr[i] << "\n";
    // }

    if (myItr.empty()) {
      //std::cout << "HORRENDOUS ERROR!  No meas on track!" << "\n";
    }
    unsigned int ilayer;
    for (ilayer = 0; ilayer < allLayers.size(); ++ilayer) {
      //std::cout <<  " allLayers in the search loop  " << allLayers[ilayer] <<  " " << myLayers[0] <<  "\n";
      if (allLayers[ilayer] == myLayers[0]) {
        myPointer = myItr[0];

        //std::cout <<  " allLayers in the search loop   allLayers[ilayer] == myLayers[0])  " << allLayers[ilayer] <<  " " << myLayers[0] <<  " myPointer " << myPointer << "\n";

        //std::cout  << "Layer " << ilayer << "  contains the first valid measurement " << "\n";
        printLayer(ilayer);

        if ((myLayers[0])->location() == GeomDetEnumerators::barrel) {
          //	  const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(myLayers[0]);
          //std::cout << " InOutConversionSeedFinder::fillClusterSeeds  **** firstHit found in Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<   "\n";
        } else {
          //const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(myLayers[0]);
          //std::cout << " InOutwardConversionSeedFinder::fillClusterSeeds  **** firstHit found in Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  "\n";
        }

        break;

      } else if (allLayers[ilayer] == myLayers[1]) {
        myPointer = myItr[1];

        //std::cout <<  " allLayers in the search loop   allLayers[ilayer] == myLayers[1])  " << allLayers[ilayer] <<  " " << myLayers[1] <<  " myPointer " << myPointer << "\n";

        //std::cout << "Layer " << ilayer << "  contains the first innermost  valid measurement " << "\n";
        if ((myLayers[1])->location() == GeomDetEnumerators::barrel) {
          //	  const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(myLayers[1]);
          //std::cout << " InOutConversionSeedFinder::fillClusterSeeds  **** 2ndHit found in Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<   "\n";
        } else {
          //const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(myLayers[1]);
          //std::cout << " InOutwardConversionSeedFinder::fillClusterSeeds  ****  2ndHitfound on forw layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  "\n";
        }

        break;
      }
    }

    if (ilayer == allLayers.size()) {
      //std::cout << "InOutConversionSeedFinder::fillClusterSeeds ERROR could not find layer on list" <<  "\n";
      return;
    }

    //PropagatorWithMaterial reversePropagator(oppositeToMomentum, 0.000511, &(*theMF_) );
    assert(myPointer);
    const FreeTrajectoryState* fts = myPointer->updatedState().freeTrajectoryState();

    //std::cout << " InOutConversionSeedFinder::fillClusterSeeds First FTS charge " << fts->charge() << " Position " << fts->position() << " momentum " << fts->momentum() << " R " << sqrt(fts->position().x()*fts->position().x() + fts->position().y()* fts->position().y() ) << " Z " << fts->position().z() << " phi " << fts->position().phi() << " fts parameters " << fts->parameters() << "\n";

    while (ilayer > 0) {
      //std::cout << " InOutConversionSeedFinder::fillClusterSeeds looking for 2nd seed from layer " << ilayer << "\n";

      //   if ( (allLayers[ilayer])->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(allLayers[ilayer]);
      //std::cout <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  "\n";
      // } else {
      //const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(allLayers[ilayer]);
      //std::cout <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() << "\n";
      //      }

      const DetLayer* previousLayer = allLayers[ilayer];
      TrajectoryStateOnSurface stateAtPreviousLayer;
      //std::cout << " InOutConversionSeedFinder::fillClusterSeeds previousLayer->surface() position before  " <<allLayers[ilayer] << " " <<  previousLayer->surface().position() << " layer location " << previousLayer->location() << "\n";
      // Propagate to the previous layer
      // The present layer is actually included in the loop so that a partner can be searched for
      // Applying the propagator to the same layer does not do any harm. It simply does nothing

      //     const Propagator& newProp=thePropagatorOppositeToMomentum_;
      //std::cout << " InOutConversionSeedFinder::fillClusterSeeds reversepropagator direction " << thePropagatorOppositeToMomentum_->propagationDirection()  << "\n";
      if (ilayer - 1 > 0) {
        if (allLayers[ilayer] == myLayers[0]) {
          //std::cout << " innermost hit R " << myPointer->recHit()->globalPosition().perp() << " Z " << myPointer->recHit()->globalPosition().z() << " phi " <<myPointer->recHit()->globalPosition().phi() << "\n";
          //std::cout << " surface R " << theTrackerGeom_->idToDet(  myPointer->recHit() ->geographicalId())->surface().position().perp() <<  " Z " <<  theTrackerGeom_->idToDet(  myPointer->recHit() ->geographicalId())->surface().position().z() << " phi " << theTrackerGeom_->idToDet(  myPointer->recHit() ->geographicalId())->surface().position().phi() << "\n";

          stateAtPreviousLayer = thePropagatorOppositeToMomentum_->propagate(
              *fts, theTrackerGeom_->idToDet(myPointer->recHit()->geographicalId())->surface());

        } else {
          stateAtPreviousLayer = thePropagatorOppositeToMomentum_->propagate(*fts, previousLayer->surface());
          //std::cout << " InOutConversionSeedFinder::fillClusterSeeds previousLayer->surface() position after " << previousLayer->surface().position() << " layer location " << previousLayer->location() <<   "\n";
        }

      } else if (ilayer - 1 == 0) {
        ////std::cout << " innermost hit R " << myPointer->recHit()->globalPosition().perp() << " Z " << myPointer->recHit()->globalPosition().z() << " phi " <<myPointer->recHit()->globalPosition().phi() << "\n";
        ////std::cout << " surface R " << theTrackerGeom_->idToDet(  myPointer->recHit() ->geographicalId())->surface().position().perp() <<  " Z " <<  theTrackerGeom_->idToDet(  myPointer->recHit() ->geographicalId())->surface().position().z() << " phi " << theTrackerGeom_->idToDet(  myPointer->recHit() ->geographicalId())->surface().position().phi() << "\n";

        //stateAtPreviousLayer= thePropagatorOppositeToMomentum_->propagate(*fts,   theTrackerGeom_->idToDet(  myPointer->recHit() ->geographicalId())->surface()   );
        stateAtPreviousLayer = thePropagatorOppositeToMomentum_->propagate(*fts, previousLayer->surface());
      }

      if (!stateAtPreviousLayer.isValid()) {
        //std::cout << "InOutConversionSeedFinder::fillClusterSeeds ERROR:could not propagate back to layer "  << ilayer << "\n";
        ////std::cout << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer " << stateAtPreviousLayer <<std:: endl;
      } else {
        //std::cout << "InOutConversionSeedFinder::fillClusterSeeds  stateAtPreviousLayer is valid.  Propagating back to layer "  << ilayer << "\n";
        //std::cout << "InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer R  " << stateAtPreviousLayer.globalPosition().perp()  << " Z " << stateAtPreviousLayer.globalPosition().z() << " phi " <<  stateAtPreviousLayer.globalPosition().phi() << "\n";

        startSeed(fts, stateAtPreviousLayer, -1, ilayer);
      }

      --ilayer;
    }

    if (ilayer == 0) {
      //     if ( (allLayers[ilayer])->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(allLayers[ilayer]);
      // //std::cout <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  "\n";
      // } else {
      //const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(allLayers[ilayer]);
      //std::cout <<  " InOutConversionSeedFinder::fillClusterSeeds  ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() << "\n";
      // }
      const DetLayer* previousLayer = allLayers[ilayer];
      TrajectoryStateOnSurface stateAtPreviousLayer;
      stateAtPreviousLayer = thePropagatorOppositeToMomentum_->propagate(*fts, previousLayer->surface());

      if (!stateAtPreviousLayer.isValid()) {
        //std::cout << "InOutConversionSeedFinder::fillClusterSeeds ERROR:could not propagate back to layer "  << ilayer << "\n";
        ////std::cout << "  InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer " << stateAtPreviousLayer <<std:: endl;
      } else {
        //std::cout << "InOutConversionSeedFinder::fillClusterSeeds  stateAtPreviousLayer is valid.  Propagating back to layer "  << ilayer << "\n";
        //std::cout << "InOutConversionSeedFinder::fillClusterSeeds stateAtPreviousLayer R  " << stateAtPreviousLayer.globalPosition().perp()  << " Z " << stateAtPreviousLayer.globalPosition().z() << " phi " <<  stateAtPreviousLayer.globalPosition().phi() << "\n";

        startSeed(fts, stateAtPreviousLayer, -1, ilayer);
      }
    }

  }  // End loop over Out In tracks
}

void InOutConversionSeedFinder::startSeed(const FreeTrajectoryState* fts,
                                          const TrajectoryStateOnSurface& stateAtPreviousLayer,
                                          int charge,
                                          int ilayer) {
  //std::cout << "InOutConversionSeedFinder::startSeed ilayer " << ilayer <<  "\n";
  // Get a list of basic clusters that are consistent with a track
  // starting at the assumed conversion point with opp. charge to the
  // inward track.  Loop over these basic clusters.
  track2Charge_ = charge * fts->charge();
  std::vector<const reco::CaloCluster*> bcVec;
  //std::cout << "InOutConversionSeedFinder::startSeed charge assumed for the in-out track  " << track2Charge_ <<  "\n";

  // Geom::Phi<float> theConvPhi( stateAtPreviousLayer.globalPosition().phi());
  //std::cout << "InOutConversionSeedFinder::startSeed  stateAtPreviousLayer phi " << stateAtPreviousLayer.globalPosition().phi() << " R " <<  stateAtPreviousLayer.globalPosition().perp() << " Z " << stateAtPreviousLayer.globalPosition().z() << "\n";

  bcVec = getSecondCaloClusters(stateAtPreviousLayer.globalPosition(), track2Charge_);

  std::vector<const reco::CaloCluster*>::iterator bcItr;
  //std::cout << "InOutConversionSeedFinder::startSeed bcVec.size " << bcVec.size() << "\n";

  // debug
  //  for(bcItr = bcVec.begin(); bcItr != bcVec.end(); ++bcItr) {
  //  //std::cout << "InOutConversionSeedFinder::startSeed list of  bc eta " << (*bcItr)->position().eta() << " phi " << (*bcItr)->position().phi() << " x " << (*bcItr)->position().x() << " y " << (*bcItr)->position().y() << " z " << (*bcItr)->position().z() << "\n";
  // }

  for (bcItr = bcVec.begin(); bcItr != bcVec.end(); ++bcItr) {
    theSecondBC_ = **bcItr;
    GlobalPoint bcPos((theSecondBC_.position()).x(), (theSecondBC_.position()).y(), (theSecondBC_.position()).z());

    //std::cout << "InOutConversionSeedFinder::startSeed for  bc position x " << bcPos.x() << " y " <<  bcPos.y() << " z  " <<  bcPos.z() << " eta " <<  bcPos.eta() << " phi " <<  bcPos.phi() << "\n";
    GlobalVector dir = stateAtPreviousLayer.globalDirection();
    GlobalPoint back1mm = stateAtPreviousLayer.globalPosition();
    //std::cout << "InOutConversionSeedFinder::startSeed   stateAtPreviousLayer.globalPosition() " << back1mm << "\n";

    back1mm -= dir.unit() * 0.1;
    //std::cout << " InOutConversionSeedFinder:::startSeed going to make the helix using back1mm " << back1mm <<"\n";
    ConversionFastHelix helix(bcPos, stateAtPreviousLayer.globalPosition(), back1mm, &(*theMF_));
    helix.stateAtVertex();

    //std::cout << " InOutConversionSeedFinder:::startSeed helix status " <<helix.isValid() << std::endl;
    if (!helix.isValid())
      continue;
    findSeeds(stateAtPreviousLayer, helix.stateAtVertex().transverseCurvature(), ilayer);
  }
}

std::vector<const reco::CaloCluster*> InOutConversionSeedFinder::getSecondCaloClusters(
    const GlobalPoint& conversionPosition, float charge) const {
  std::vector<const reco::CaloCluster*> result;

  Geom::Phi<float> convPhi(conversionPosition.phi());

  for (auto const& bc : *bcCollection_) {
    Geom::Phi<float> bcPhi(bc.position().phi());

    // Require phi of cluster to be consistent with the conversion position and the track charge

    if (std::abs(bcPhi - convPhi) < .5 &&
        ((charge < 0 && bcPhi - convPhi > -.5) || (charge > 0 && bcPhi - convPhi < .5)))
      result.push_back(&bc);
  }

  return result;
}

void InOutConversionSeedFinder::findSeeds(const TrajectoryStateOnSurface& startingState,
                                          float transverseCurvature,
                                          unsigned int startingLayer) {
  std::vector<const DetLayer*> allLayers = layerList();
  //std::cout << "InOutConversionSeedFinder::findSeeds starting forward propagation from  startingLayer " << startingLayer <<  "\n";

  // create error matrix
  AlgebraicSymMatrix55 m = AlgebraicMatrixID();
  m(0, 0) = 0.1;
  m(1, 1) = 0.0001;
  m(2, 2) = 0.0001;
  m(3, 3) = 0.0001;
  m(4, 4) = 0.001;

  // Make an FTS consistent with the start point, start direction and curvature
  FreeTrajectoryState fts(
      GlobalTrajectoryParameters(
          startingState.globalPosition(), startingState.globalDirection(), double(transverseCurvature), 0, &(*theMF_)),
      CurvilinearTrajectoryError(m));
  if (fts.momentum().mag2() == 0) {
    edm::LogWarning("FailedToInitiateSeeding")
        << " initial FTS has a zero momentum, probably because of the zero field.  ";
    return;
  }
  //std::cout << " InOutConversionSeedFinder::findSeeds startingState R "<< startingState.globalPosition().perp() << " Z " << startingState.globalPosition().z() << " phi " <<  startingState.globalPosition().phi() <<  " position " << startingState.globalPosition() << "\n";
  //std::cout << " InOutConversionSeedFinder::findSeeds Initial FTS charge " << fts.charge() << " curvature " <<  transverseCurvature << "\n";
  //std::cout << " InOutConversionSeedFinder::findSeeds Initial FTS parameters " << fts <<  "\n";

  //float dphi = 0.01;
  float dphi = 0.03;
  float zrange = 5.;
  for (unsigned int ilayer = startingLayer; ilayer <= startingLayer + 1 && (ilayer < allLayers.size() - 2); ++ilayer) {
    const DetLayer* layer = allLayers[ilayer];

    ///// debug
    //    if ( layer->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(layer);
    // //std::cout << "InOutConversionSeedFinder::findSeeds  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  "\n";
    // } else {
    // const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(layer);
    ////std::cout << "InOutConversionSeedFinder::findSeeds  ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  "\n";
    // }
    // // end debug

    MeasurementEstimator* newEstimator = nullptr;
    if (layer->location() == GeomDetEnumerators::barrel) {
      //std::cout << "InOutConversionSeedFinder::findSeeds Barrel ilayer " << ilayer <<  "\n";
      newEstimator = new ConversionBarrelEstimator(-dphi, dphi, -zrange, zrange);
    } else {
      //std::cout << "InOutConversionSeedFinder::findSeeds Forward  ilayer " << ilayer <<  "\n";
      newEstimator = new ConversionForwardEstimator(-dphi, dphi, 15.);
    }

    theFirstMeasurements_.clear();
    // Get measurements compatible with the FTS and Estimator
    TSOS tsos(fts, layer->surface());

    //std::cout << "InOutConversionSeedFinder::findSeed propagationDirection " << int(thePropagatorAlongMomentum_->propagationDirection() ) << "\n";
    /// Rememeber that this alwyas give back at least one dummy-innvalid it which prevents from everything getting stopped
    LayerMeasurements theLayerMeasurements_(*this->getMeasurementTracker(), *theTrackerData_);

    theFirstMeasurements_ =
        theLayerMeasurements_.measurements(*layer, tsos, *thePropagatorAlongMomentum_, *newEstimator);

    delete newEstimator;
    //std::cout <<  "InOutConversionSeedFinder::findSeeds  Found " << theFirstMeasurements_.size() << " first hits" << "\n";

    if (theFirstMeasurements_.size() ==
        1) {  // only dummy hit found: start finding the seed from the innermost hit of the OutIn track

      GlobalPoint bcPos((theSecondBC_.position()).x(), (theSecondBC_.position()).y(), (theSecondBC_.position()).z());
      GlobalVector dir = startingState.globalDirection();
      GlobalPoint back1mm = myPointer->recHit()->globalPosition();

      back1mm -= dir.unit() * 0.1;
      //std::cout << " InOutConversionSeedFinder:::findSeeds going to make the helix using back1mm " << back1mm << "\n";
      ConversionFastHelix helix(bcPos, myPointer->recHit()->globalPosition(), back1mm, &(*theMF_));

      helix.stateAtVertex();
      //std::cout << " InOutConversionSeedFinder:::findSeeds helix status " <<helix.isValid() << std::endl;
      if (!helix.isValid())
        continue;

      track2InitialMomentum_ = helix.stateAtVertex().momentum();

      // Make a new FTS
      FreeTrajectoryState newfts(GlobalTrajectoryParameters(myPointer->recHit()->globalPosition(),
                                                            startingState.globalDirection(),
                                                            helix.stateAtVertex().transverseCurvature(),
                                                            0,
                                                            &(*theMF_)),
                                 CurvilinearTrajectoryError(m));

      completeSeed(*myPointer, newfts, thePropagatorAlongMomentum_, ilayer + 1);
      completeSeed(*myPointer, newfts, thePropagatorAlongMomentum_, ilayer + 2);

    } else {
      //Loop over compatible hits
      for (std::vector<TrajectoryMeasurement>::iterator tmItr = theFirstMeasurements_.begin();
           tmItr != theFirstMeasurements_.end();
           ++tmItr) {
        if (tmItr->recHit()->isValid()) {
          // Make a new helix as in fillClusterSeeds() but using the hit position
          //std::cout << "InOutConversionSeedFinder::findSeeds hit  R " << tmItr->recHit()->globalPosition().perp() << " Z " <<  tmItr->recHit()->globalPosition().z() << " " <<  tmItr->recHit()->globalPosition() << "\n";
          GlobalPoint bcPos(
              (theSecondBC_.position()).x(), (theSecondBC_.position()).y(), (theSecondBC_.position()).z());
          GlobalVector dir = startingState.globalDirection();
          GlobalPoint back1mm = tmItr->recHit()->globalPosition();

          back1mm -= dir.unit() * 0.1;
          //std::cout << " InOutConversionSeedFinder:::findSeeds going to make the helix using back1mm " << back1mm << "\n";
          ConversionFastHelix helix(bcPos, tmItr->recHit()->globalPosition(), back1mm, &(*theMF_));

          helix.stateAtVertex();
          //std::cout << " InOutConversionSeedFinder:::findSeeds helix status " <<helix.isValid() << std::endl;
          if (!helix.isValid())
            continue;

          track2InitialMomentum_ = helix.stateAtVertex().momentum();

          //std::cout << "InOutConversionSeedFinder::findSeeds Updated estimatedPt = " << helix.stateAtVertex().momentum().perp()  << " curvature "  << helix.stateAtVertex().transverseCurvature() << "\n";
          //     << ", bcet = " << theBc->Et()
          //     << ", estimatedPt/bcet = " << estimatedPt/theBc->Et() << endl;

          // Make a new FTS
          FreeTrajectoryState newfts(GlobalTrajectoryParameters(tmItr->recHit()->globalPosition(),
                                                                startingState.globalDirection(),
                                                                helix.stateAtVertex().transverseCurvature(),
                                                                0,
                                                                &(*theMF_)),
                                     CurvilinearTrajectoryError(m));

          //std::cout <<  "InOutConversionSeedFinder::findSeeds  new FTS charge " << newfts.charge() << "\n";

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

          completeSeed(*tmItr, newfts, thePropagatorAlongMomentum_, ilayer + 1);
          completeSeed(*tmItr, newfts, thePropagatorAlongMomentum_, ilayer + 2);
        }
      }
    }
  }
}

void InOutConversionSeedFinder::completeSeed(const TrajectoryMeasurement& m1,
                                             const FreeTrajectoryState& fts,
                                             const Propagator* propagator,
                                             int ilayer) {
  //std::cout<<  "InOutConversionSeedFinder::completeSeed ilayer " << ilayer <<  "\n";
  // A seed is made from 2 Trajectory Measuremennts.  The 1st is the input
  // argument m1.  This routine looks for the 2nd measurement in layer ilayer
  // Begin by making a new much stricter MeasurementEstimator based on the
  // position errors of the 1st hit.
  printLayer(ilayer);

  MeasurementEstimator* newEstimator;
  std::vector<const DetLayer*> allLayers = layerList();
  const DetLayer* layer = allLayers[ilayer];

  ///// debug
  //  if ( layer->location() == GeomDetEnumerators::barrel ) {const BarrelDetLayer * barrelLayer = dynamic_cast<const BarrelDetLayer*>(layer);
  // //std::cout << "InOutConversionSeedFinder::completeSeed  ****  Barrel on layer " << ilayer  << " R= " << barrelLayer->specificSurface().radius() <<  "\n";
  // } else {
  // const ForwardDetLayer * forwardLayer = dynamic_cast<const ForwardDetLayer*>(layer);
  //  //std::cout << "InOutConversionSeedFinder::completeSeed ****  Forw on layer " << ilayer  << " Z= " << forwardLayer->specificSurface().position().z() <<  "\n";
  ///  }
  //// end debug

  if (layer->location() == GeomDetEnumerators::barrel) {
    float dz = sqrt(the2ndHitdznSigma_ * the2ndHitdznSigma_ * m1.recHit()->globalPositionError().czz() +
                    the2ndHitdzConst_ * the2ndHitdzConst_);
    newEstimator = new ConversionBarrelEstimator(-the2ndHitdphi_, the2ndHitdphi_, -dz, dz);

  } else {
    float m1dr = sqrt(m1.recHit()->localPositionError().yy());
    float dr = sqrt(the2ndHitdznSigma_ * the2ndHitdznSigma_ * m1dr * m1dr + the2ndHitdzConst_ * the2ndHitdznSigma_);

    newEstimator = new ConversionForwardEstimator(-the2ndHitdphi_, the2ndHitdphi_, dr);
  }

  //std::cout << "InOutConversionSeedFinder::completeSeed fts For the TSOS " << fts << "\n";

  TSOS tsos(fts, layer->surface());

  if (!tsos.isValid()) {
    //std::cout  << "InOutConversionSeedFinder::completeSeed TSOS is not valid " <<  "\n";
  }

  //std::cout << "InOutConversionSeedFinder::completeSeed TSOS " << tsos << "\n";
  //std::cout << "InOutConversionSeedFinder::completeSeed propagationDirection  " << int(propagator->propagationDirection() ) << "\n";
  //std::cout << "InOutConversionSeedFinder::completeSeed pointer to estimator " << newEstimator << "\n";

  LayerMeasurements theLayerMeasurements_(*this->getMeasurementTracker(), *theTrackerData_);
  std::vector<TrajectoryMeasurement> measurements =
      theLayerMeasurements_.measurements(*layer, tsos, *propagator, *newEstimator);
  //std::cout << "InOutConversionSeedFinder::completeSeed Found " << measurements.size() << " second hits " <<  "\n";
  delete newEstimator;

  for (unsigned int i = 0; i < measurements.size(); ++i) {
    if (measurements[i].recHit()->isValid()) {
      createSeed(m1, measurements[i]);
    }
  }
}

void InOutConversionSeedFinder::createSeed(const TrajectoryMeasurement& m1, const TrajectoryMeasurement& m2) {
  //std::cout << "InOutConversionSeedFinder::createSeed " << "\n";

  if (m1.predictedState().isValid()) {
    GlobalTrajectoryParameters newgtp(m1.recHit()->globalPosition(), track2InitialMomentum_, track2Charge_, &(*theMF_));
    CurvilinearTrajectoryError errors = m1.predictedState().curvilinearError();
    FreeTrajectoryState fts(newgtp, errors);

    TrajectoryStateOnSurface state1 = thePropagatorAlongMomentum_->propagate(fts, m1.recHit()->det()->surface());

    /*
   //std::cout << "hit surface " <<  m1.recHit()->det()->surface().position() << "\n";
   //std::cout << "prop to " << typeid( m1.recHit()->det()->surface() ).name() <<"\n";
   //std::cout << "prop to first hit " << state1 << "\n"; 
   //std::cout << "update to " <<  m1.recHit()->globalPosition() << "\n";
  */

    if (state1.isValid()) {
      TrajectoryStateOnSurface updatedState1 = theUpdator_.update(state1, *m1.recHit());

      if (updatedState1.isValid()) {
        TrajectoryStateOnSurface state2 =
            thePropagatorAlongMomentum_->propagate(*updatedState1.freeTrajectoryState(), m2.recHit()->det()->surface());

        if (state2.isValid()) {
          TrajectoryStateOnSurface updatedState2 = theUpdator_.update(state2, *m2.recHit());
          TrajectoryMeasurement meas1(state1, updatedState1, m1.recHit(), m1.estimate(), m1.layer());
          TrajectoryMeasurement meas2(state2, updatedState2, m2.recHit(), m2.estimate(), m2.layer());

          edm::OwnVector<TrackingRecHit> myHits;
          myHits.push_back(meas1.recHit()->hit()->clone());
          myHits.push_back(meas2.recHit()->hit()->clone());

          //std::cout << "InOutConversionSeedFinder::createSeed new seed " << "\n";
          if (nSeedsPerInputTrack_ >= maxNumberOfInOutSeedsPerInputTrack_)
            return;

          PTrajectoryStateOnDet const& ptsod =
              trajectoryStateTransform::persistentState(state2, meas2.recHit()->hit()->geographicalId().rawId());
          //std::cout << "  InOutConversionSeedFinder::createSeed New seed parameters " << state2 << "\n";

          theSeeds_.push_back(TrajectorySeed(ptsod, myHits, alongMomentum));
          nSeedsPerInputTrack_++;

          //std::cout << "InOutConversionSeedFinder::createSeed New seed hit 1 R " << m1.recHit()->globalPosition().perp() << "\n";
          //std::cout << "InOutConversionSeedFinder::createSeed New seed hit 2 R " << m2.recHit()->globalPosition().perp() << "\n";
        }
      }
    }
  }
}
