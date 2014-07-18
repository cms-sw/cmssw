#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/Math/interface/Vector.h" 
#include "DataFormats/Math/interface/Error.h" 
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h" 
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h" 
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectory.h"

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayFitter.h"

/**
   by Edmund Widl, see CMS NOTE-2007/032.
   extension for BreakPoints or BrokenLines by Claus Kleinwort
 */

class TwoBodyDecayTrajectoryFactory : public TrajectoryFactoryBase
{
public:

  typedef TwoBodyDecayVirtualMeasurement VirtualMeasurement;
  typedef TwoBodyDecayTrajectoryState::TsosContainer TsosContainer;
  typedef TwoBodyDecayTrajectory::ConstRecHitCollection ConstRecHitCollection;

  TwoBodyDecayTrajectoryFactory(const edm::ParameterSet &config);
  ~TwoBodyDecayTrajectoryFactory();

  /// Produce the trajectories.
  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const reco::BeamSpot &beamSpot) const override;

  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const ExternalPredictionCollection &external,
							   const reco::BeamSpot &beamSpot) const override;

  virtual TwoBodyDecayTrajectoryFactory* clone() const override { return new TwoBodyDecayTrajectoryFactory(*this); }

protected:
  const ReferenceTrajectoryCollection constructTrajectories(const ConstTrajTrackPairCollection &tracks,
							    const TwoBodyDecay &tbd,
							    const MagneticField *magField,
							    const reco::BeamSpot &beamSpot,
							    bool setParameterErrors) const;

  bool match(const TrajectoryStateOnSurface &state,
	     const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;
    
  TwoBodyDecayFitter* theFitter;

  double thePrimaryMass;
  double thePrimaryWidth;
  double theSecondaryMass;

  double theNSigmaCutValue;
  double theChi2CutValue;
  bool theUseRefittedStateFlag;
  bool theConstructTsosWithErrorsFlag;

};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TwoBodyDecayTrajectoryFactory::TwoBodyDecayTrajectoryFactory( const edm::ParameterSet& config )
  : TrajectoryFactoryBase( config ),
    theFitter( new TwoBodyDecayFitter( config ) )
{
  const edm::ParameterSet ppc = config.getParameter< edm::ParameterSet >( "ParticleProperties" );
  thePrimaryMass = ppc.getParameter< double >( "PrimaryMass" );
  thePrimaryWidth = ppc.getParameter< double >( "PrimaryWidth" );
  theSecondaryMass = ppc.getParameter< double >( "SecondaryMass" );

  theNSigmaCutValue = config.getParameter< double >( "NSigmaCut" );
  theChi2CutValue = config.getParameter< double >( "Chi2Cut" );
  theUseRefittedStateFlag = config.getParameter< bool >( "UseRefittedState" );
  theConstructTsosWithErrorsFlag = config.getParameter< bool >( "ConstructTsosWithErrors" );
}

TwoBodyDecayTrajectoryFactory::~TwoBodyDecayTrajectoryFactory()
{
  delete theFitter;
}

const TrajectoryFactoryBase::ReferenceTrajectoryCollection
TwoBodyDecayTrajectoryFactory::trajectories(const edm::EventSetup &setup,
					    const ConstTrajTrackPairCollection &tracks,
					    const reco::BeamSpot &beamSpot) const
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
    VirtualMeasurement vm( thePrimaryMass, thePrimaryWidth, theSecondaryMass, beamSpot );
    TwoBodyDecay tbd = theFitter->estimate( transientTracks, vm );

    if ( !tbd.isValid() || ( tbd.chi2() > theChi2CutValue ) )
    {
      trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory() ) );
      return trajectories;
    }

    return constructTrajectories( tracks, tbd, magneticField.product(), beamSpot, false );
  }
  else
  {
    edm::LogInfo( "ReferenceTrajectories" ) << "@SUB=TwoBodyDecayTrajectoryFactory::trajectories"
					    << "Need 2 tracks, got " << tracks.size() << ".\n";
  }

  return trajectories;
}


const TrajectoryFactoryBase::ReferenceTrajectoryCollection
TwoBodyDecayTrajectoryFactory::trajectories(const edm::EventSetup &setup,
					    const ConstTrajTrackPairCollection &tracks,
					    const ExternalPredictionCollection &external,
					    const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );

  if ( tracks.size() == 2 && external.size() == 2 )
  {
    if ( external[0].isValid() && external[1].isValid() ) // Include external estimates
    {
      // produce transient tracks from persistent tracks
      std::vector< reco::TransientTrack > transientTracks( 2 );

      transientTracks[0] = reco::TransientTrack( *tracks[0].second, magneticField.product() );
      transientTracks[0].setES( setup );

      transientTracks[1] = reco::TransientTrack( *tracks[1].second, magneticField.product() );
      transientTracks[1].setES( setup );

      // estimate the decay parameters. the transient tracks are not really associated to the
      // the external tsos, but this is o.k., because the only information retrieved from them
      // is the magnetic field.
      VirtualMeasurement vm( thePrimaryMass, thePrimaryWidth, theSecondaryMass, beamSpot );
      TwoBodyDecay tbd = theFitter->estimate( transientTracks, external, vm );

      if ( !tbd.isValid()  || ( tbd.chi2() > theChi2CutValue ) )
      {
	trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory() ) );
	return trajectories;
      }

      return constructTrajectories( tracks, tbd, magneticField.product(), beamSpot, true );
    }
    else
    {
      // Return without external estimate
      trajectories = this->trajectories(setup, tracks, beamSpot);
    }
  }
  else
  {
    edm::LogInfo( "ReferenceTrajectories" ) << "@SUB=TwoBodyDecayTrajectoryFactory::trajectories"
					    << "Need 2 tracks, got " << tracks.size() << ".\n";
  }

  return trajectories;
}


const TwoBodyDecayTrajectoryFactory::ReferenceTrajectoryCollection
TwoBodyDecayTrajectoryFactory::constructTrajectories( const ConstTrajTrackPairCollection& tracks,
						      const TwoBodyDecay& tbd,
						      const MagneticField* magField,
						      const reco::BeamSpot &beamSpot,
						      bool setParameterErrors ) const
{
  ReferenceTrajectoryCollection trajectories;

  // get innermost valid trajectory state and hits from the tracks
  TrajectoryInput input1 = this->innermostStateAndRecHits( tracks[0] );
  TrajectoryInput input2 = this->innermostStateAndRecHits( tracks[1] );

  if ( !( input1.first.isValid() && input2.first.isValid() ) ) return trajectories;

  // produce TwoBodyDecayTrajectoryState (input for TwoBodyDecayTrajectory)
  TsosContainer tsos( input1.first, input2.first );
  ConstRecHitCollection recHits( input1.second, input2.second );
  TwoBodyDecayTrajectoryState trajectoryState( tsos, tbd, theSecondaryMass, magField );

  if ( !trajectoryState.isValid() )
  {
    trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory() ) );
    return trajectories;
  }

  // always use the refitted trajectory state for matching
  // FIXME FIXME CLONE
  //TrackingRecHit::ConstRecHitPointer updatedRecHit1( recHits.first.front()->clone( tsos.first ) );
  // TrackingRecHit::ConstRecHitPointer updatedRecHit2( recHits.second.front()->clone( tsos.second ) );

  TrackingRecHit::ConstRecHitPointer updatedRecHit1( recHits.first.front() );
  TrackingRecHit::ConstRecHitPointer updatedRecHit2( recHits.second.front() );


  bool valid1 = match( trajectoryState.trajectoryStates( true ).first,
		       updatedRecHit1 );

  bool valid2 = match( trajectoryState.trajectoryStates( true ).second,
		       updatedRecHit2 );

  if ( !valid1 || !valid2 )
  {
    trajectories.push_back( ReferenceTrajectoryPtr( new TwoBodyDecayTrajectory() ) );
    return trajectories;
  }

  // set the flag for reversing the RecHits to false, since they are already in the correct order.
  TwoBodyDecayTrajectory* result = new TwoBodyDecayTrajectory( trajectoryState, recHits, magField,
							       materialEffects(), propagationDirection(),
							       false, beamSpot, theUseRefittedStateFlag,
							       theConstructTsosWithErrorsFlag );
  if ( setParameterErrors && tbd.hasError() ) result->setParameterErrors( tbd.covariance() );
  trajectories.push_back( ReferenceTrajectoryPtr( result ) );
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


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, TwoBodyDecayTrajectoryFactory, "TwoBodyDecayTrajectoryFactory" );
