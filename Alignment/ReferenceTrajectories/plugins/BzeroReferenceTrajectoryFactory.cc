// Do not include .h from plugin directory, but locally:
#include "BzeroReferenceTrajectoryFactory.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 


BzeroReferenceTrajectoryFactory::BzeroReferenceTrajectoryFactory( const edm::ParameterSet & config ) :
  TrajectoryFactoryBase( config )
{
  theMass = config.getParameter< double >( "ParticleMass" );
  theMomentumEstimate = config.getParameter< double >( "MomentumEstimate" );
}

 
BzeroReferenceTrajectoryFactory::~BzeroReferenceTrajectoryFactory( void ) {}


const BzeroReferenceTrajectoryFactory::ReferenceTrajectoryCollection
BzeroReferenceTrajectoryFactory::trajectories( const edm::EventSetup & setup,
					       const ConstTrajTrackPairCollection & tracks ) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );

  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();

  while ( itTracks != tracks.end() )
  { 
    TrajectoryInput input = this->innermostStateAndRecHits( *itTracks );
    // set the flag for reversing the RecHits to false, since they are already in the correct order.
    trajectories.push_back( ReferenceTrajectoryPtr( new BzeroReferenceTrajectory( input.first, input.second, 
										  false, magneticField.product(),
										  theMaterialEffects, theMass,
										  theMomentumEstimate ) ) );
    ++itTracks;
  }

  return trajectories;
}


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, BzeroReferenceTrajectoryFactory, "BzeroReferenceTrajectoryFactory" );
