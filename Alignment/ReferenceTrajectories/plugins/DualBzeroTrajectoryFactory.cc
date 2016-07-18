#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include <algorithm>

#include "Alignment/ReferenceTrajectories/interface/DualBzeroReferenceTrajectory.h" 

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

/// A factory that produces instances of class ReferenceTrajectory from a given TrajTrackPairCollection.


class DualBzeroTrajectoryFactory : public TrajectoryFactoryBase
{
public:
  DualBzeroTrajectoryFactory(const edm::ParameterSet &config);
  virtual ~DualBzeroTrajectoryFactory();

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							    const reco::BeamSpot &beamSpot) const override;

  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const ExternalPredictionCollection &external,
							   const reco::BeamSpot &beamSpot) const override;

  virtual DualBzeroTrajectoryFactory* clone() const override { return new DualBzeroTrajectoryFactory(*this); }

protected:
  struct DualBzeroTrajectoryInput
  {
    TrajectoryStateOnSurface refTsos;
    TransientTrackingRecHit::ConstRecHitContainer fwdRecHits;
    TransientTrackingRecHit::ConstRecHitContainer bwdRecHits;
  };

  const DualBzeroTrajectoryInput referenceStateAndRecHits(const ConstTrajTrackPair &track) const;

  const TrajectoryStateOnSurface propagateExternal(const TrajectoryStateOnSurface &external,
						   const Surface &surface,
						   const MagneticField *magField) const;

  double theMass;
  double theMomentumEstimate;
  
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DualBzeroTrajectoryFactory::DualBzeroTrajectoryFactory( const edm::ParameterSet & config ) :
  TrajectoryFactoryBase( config )
{
  theMass = config.getParameter< double >( "ParticleMass" );
  theMomentumEstimate = config.getParameter< double >( "MomentumEstimate" );
}

 
DualBzeroTrajectoryFactory::~DualBzeroTrajectoryFactory( void ) {}


const DualBzeroTrajectoryFactory::ReferenceTrajectoryCollection
DualBzeroTrajectoryFactory::trajectories(const edm::EventSetup  &setup,
					 const ConstTrajTrackPairCollection &tracks,
					 const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );

  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();

  while ( itTracks != tracks.end() )
  { 
    const DualBzeroTrajectoryInput input = this->referenceStateAndRecHits( *itTracks );
    // Check input: If all hits were rejected, the TSOS is initialized as invalid.
    if ( input.refTsos.isValid() )
    {
      ReferenceTrajectoryBase::Config config(materialEffects(), propagationDirection(),
                                             theMass, theMomentumEstimate);
      config.useBeamSpot = useBeamSpot_;
      config.includeAPEs = includeAPEs_;
      ReferenceTrajectoryPtr ptr(new DualBzeroReferenceTrajectory(input.refTsos,
                                                                  input.fwdRecHits,
                                                                  input.bwdRecHits,
                                                                  magneticField.product(),
                                                                  beamSpot,
                                                                  config));
      trajectories.push_back( ptr );
    }

    ++itTracks;
  }

  return trajectories;
}

const DualBzeroTrajectoryFactory::ReferenceTrajectoryCollection
DualBzeroTrajectoryFactory::trajectories(const edm::EventSetup &setup,
					 const ConstTrajTrackPairCollection &tracks,
					 const ExternalPredictionCollection &external,
					 const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;

  if ( tracks.size() != external.size() )
  {
    edm::LogInfo("ReferenceTrajectories") << "@SUB=DualBzeroTrajectoryFactory::trajectories"
					  << "Inconsistent input:\n"
					  << "\tnumber of tracks = " << tracks.size()
					  << "\tnumber of external predictions = " << external.size();
    return trajectories;
  }

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );

  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();
  ExternalPredictionCollection::const_iterator itExternal = external.begin();

  while ( itTracks != tracks.end() )
  {
    const DualBzeroTrajectoryInput input = referenceStateAndRecHits( *itTracks );
    // Check input: If all hits were rejected, the TSOS is initialized as invalid.
    if ( input.refTsos.isValid() )
    {
      if ( (*itExternal).isValid() )
      {
	TrajectoryStateOnSurface propExternal =
	  propagateExternal( *itExternal, input.refTsos.surface(), magneticField.product() );

	if ( !propExternal.isValid() ) continue;

        ReferenceTrajectoryBase::Config config(materialEffects(), propagationDirection(),
                                               theMass, theMomentumEstimate);
        config.useBeamSpot = useBeamSpot_;
        config.includeAPEs = includeAPEs_;
        ReferenceTrajectoryPtr ptr(new DualBzeroReferenceTrajectory(propExternal,
                                                                    input.fwdRecHits,
                                                                    input.bwdRecHits,
                                                                    magneticField.product(),
                                                                    beamSpot,
                                                                    config));

	AlgebraicSymMatrix externalParamErrors( asHepMatrix<5>( propExternal.localError().matrix() ) );
	ptr->setParameterErrors( externalParamErrors.sub( 2, 5 ) );
	trajectories.push_back( ptr );
      }
      else
      {
        ReferenceTrajectoryBase::Config config(materialEffects(), propagationDirection(),
                                               theMass, theMomentumEstimate);
        config.useBeamSpot = useBeamSpot_;
        config.includeAPEs = includeAPEs_;
        ReferenceTrajectoryPtr ptr(new DualBzeroReferenceTrajectory(input.refTsos,
                                                                    input.fwdRecHits,
                                                                    input.bwdRecHits,
                                                                    magneticField.product(),
                                                                    beamSpot,
                                                                    config));

	trajectories.push_back( ptr );
      }
    }

    ++itTracks;
    ++itExternal;
  }

  return trajectories;
}


const DualBzeroTrajectoryFactory::DualBzeroTrajectoryInput
DualBzeroTrajectoryFactory::referenceStateAndRecHits( const ConstTrajTrackPair& track ) const
{
  DualBzeroTrajectoryInput input;
 
  // get the trajectory measurements in the correct order, i.e. reverse if needed
  Trajectory::DataContainer allTrajMeas = this->orderedTrajectoryMeasurements( *track.first );
  Trajectory::DataContainer usedTrajMeas;
  Trajectory::DataContainer::iterator itM;
  // get all relevant trajectory measurements
  for ( itM = allTrajMeas.begin(); itM != allTrajMeas.end(); itM++ )
  {
    if ( useRecHit( ( *itM ).recHit() ) ) usedTrajMeas.push_back( *itM );
  }

  unsigned int iMeas = 0;
  unsigned int nMeas = usedTrajMeas.size();
  unsigned int nRefStateMeas = nMeas/2;
  // get the valid RecHits
  for ( itM = usedTrajMeas.begin(); itM != usedTrajMeas.end(); itM++, iMeas++ )
  {
    TransientTrackingRecHit::ConstRecHitPointer aRecHit = ( *itM ).recHit();

    if ( iMeas < nRefStateMeas ) {
      input.bwdRecHits.push_back( aRecHit );
    } else if ( iMeas > nRefStateMeas ) {
      input.fwdRecHits.push_back( aRecHit );
    } else { // iMeas == nRefStateMeas
      if ( ( *itM ).updatedState().isValid() )
      {
	input.refTsos = ( *itM ).updatedState();
	input.bwdRecHits.push_back( aRecHit );
	input.fwdRecHits.push_back( aRecHit );
      } else {
	// if the tsos of the middle hit is not valid, try the next one ...
	nRefStateMeas++;
	input.bwdRecHits.push_back( aRecHit );
      }
    }
  }

  // bring input.fwdRecHits into correct order
  std::reverse( input.bwdRecHits.begin(), input.bwdRecHits.end() );

  return input;
}

const TrajectoryStateOnSurface
DualBzeroTrajectoryFactory::propagateExternal( const TrajectoryStateOnSurface& external,
					  const Surface& surface,
					  const MagneticField* magField ) const
{
  AnalyticalPropagator propagator( magField, anyDirection );
  const std::pair< TrajectoryStateOnSurface, double > tsosWithPath =
    propagator.propagateWithPath( external, surface );
  return tsosWithPath.first;
}


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, DualBzeroTrajectoryFactory, "DualBzeroTrajectoryFactory" );
