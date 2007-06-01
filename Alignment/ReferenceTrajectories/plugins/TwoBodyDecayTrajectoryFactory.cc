#include "Alignment/ReferenceTrajectories/plugins/TwoBodyDecayTrajectoryFactory.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


TwoBodyDecayTrajectoryFactory::TwoBodyDecayTrajectoryFactory( const edm::ParameterSet& config )
  : TrajectoryFactoryBase( config ),
    theFitter( new TwoBodyDecayFitter( config ) ),
    theNSigmaCutValue( config.getParameter< double >( "NSigmaCut" ) ),
    theUseRefittedStateFlag( config.getParameter< bool >( "UseRefittedState" ) ),
    theConstructTsosWithErrorsFlag( config.getParameter< bool >( "ConstructTsosWithErrors" ) )
{
  produceVirtualMeasurement( config );
}


const TrajectoryFactoryBase::ReferenceTrajectoryCollection
TwoBodyDecayTrajectoryFactory::trajectories( const edm::EventSetup& setup,
					     const ConstTrajTrackPairCollection& tracks ) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );

  if ( tracks.size() == 2 )
  {
    // produce transient tracks from persistent tracks
    std::vector< reco::TransientTrack > transientTracks( 2 );

    transientTracks[0] = reco::TransientTrack( *tracks[0].second, magneticField.product() );
    transientTracks[0].setES( setup );

    transientTracks[1] = reco::TransientTrack( *tracks[1].second, magneticField.product() );
    transientTracks[1].setES( setup );

    // estimate the decay parameters
    TwoBodyDecay tbd = theFitter->estimate( transientTracks, theVM );

    if ( !tbd.isValid() )
    {
      trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory() ) );
      return trajectories;
    }

    // get innermost valid trajectory state and hits from the tracks
    TrajectoryInput input1 = innermostStateAndRecHits( tracks[0] );
    TrajectoryInput input2 = innermostStateAndRecHits( tracks[1] );

    // produce TwoBodyDecayTrajectoryState (input for TwoBodyDecayTrajectory)
    TsosContainer tsos( input1.first, input2.first );
    ConstRecHitCollection recHits( input1.second, input2.second );
    TwoBodyDecayTrajectoryState trajectoryState( tsos, tbd, theVM.secondaryMass(), magneticField.product() );

    if ( !trajectoryState.isValid() )
    {
      trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory() ) );
      return trajectories;
    }

    // always use the refitted trajectory state for matching
    TransientTrackingRecHit::ConstRecHitPointer updatedRecHit1( recHits.first.front()->clone( tsos.first ) );
    bool valid1 = match( trajectoryState.trajectoryStates( true ).first,
			 updatedRecHit1 );

    TransientTrackingRecHit::ConstRecHitPointer updatedRecHit2( recHits.second.front()->clone( tsos.second ) );
    bool valid2 = match( trajectoryState.trajectoryStates( true ).second,
			 updatedRecHit2 );

    if ( !valid1 || !valid2 )
    {
      trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory() ) );
      return trajectories;
    }

    // set the flag for reversing the RecHits to false, since they are already in the correct order.
    trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory( trajectoryState, recHits,
										magneticField.product(),
										theMaterialEffects, false,
										theUseRefittedStateFlag,
										theConstructTsosWithErrorsFlag ) ) );
  }
  else
  {
    edm::LogInfo( "Alignment" ) << "@SUB=TwoBodyDecayTrajectoryFactory::trajectories"
				<< "Need 2 tracks, got " << tracks.size() << ".\n";
  }

  return trajectories;
}


bool TwoBodyDecayTrajectoryFactory::match( const TrajectoryStateOnSurface& state,
					   const TransientTrackingRecHit::ConstRecHitPointer& recHit ) const
{
  LocalPoint lp1 = state.localPosition();
  LocalPoint lp2 = recHit->localPosition();

  double deltaX = lp1.x() - lp2.x();
  double deltaY = lp1.y() - lp2.y();

  LocalError le = recHit->localPositionError();

  double varX = le.xx();
  double varY = le.yy();

  AlignmentPositionError* gape = recHit->det()->alignmentPositionError();
  if ( gape )
  {
    ErrorFrameTransformer eft;
    LocalError lape = eft.transform( gape->globalError(), recHit->det()->surface() );

    varX += lape.xx();
    varY += lape.yy();
  }

  return ( ( fabs(deltaX)/sqrt(varX) < theNSigmaCutValue ) && ( fabs(deltaY)/sqrt(varY) < theNSigmaCutValue ) );
}


void TwoBodyDecayTrajectoryFactory::produceVirtualMeasurement( const edm::ParameterSet& config )
{
  const edm::ParameterSet bsc = config.getParameter< edm::ParameterSet >( "BeamSpot" );
  const edm::ParameterSet ppc = config.getParameter< edm::ParameterSet >( "ParticleProperties" );

  theBeamSpot = GlobalPoint( bsc.getParameter< double >( "MeanX" ),
			     bsc.getParameter< double >( "MeanY" ),
			     bsc.getParameter< double >( "MeanZ" ) );

  theBeamSpotError = GlobalError( bsc.getParameter< double >( "VarXX" ),
				  bsc.getParameter< double >( "VarXY" ),
				  bsc.getParameter< double >( "VarYY" ),
				  bsc.getParameter< double >( "VarXZ" ),
				  bsc.getParameter< double >( "VarYZ" ),
				  bsc.getParameter< double >( "VarZZ" ) );

  theVM = VirtualMeasurement( ppc.getParameter< double >( "PrimaryMass" ),
			      ppc.getParameter< double >( "PrimaryWidth" ),
			      ppc.getParameter< double >( "SecondaryMass" ),
			      theBeamSpot, theBeamSpotError );
  return;
}


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, TwoBodyDecayTrajectoryFactory, "TwoBodyDecayTrajectoryFactory" );
