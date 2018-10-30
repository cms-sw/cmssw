#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include <TString.h>
#include <TObjArray.h>

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

/// A factory that can combine the functionality of several 'trajectory factories'. At construction
/// time, it is given an ordered list of what kinds factories it should use. When called, all the
/// factories are called one after each other, 
///  - either until one of them gives a result
///  - or until all factories are really called.
/// This is determined by the useAllFactories flag.
///
/// Example: 
/// Combine TwoBodyDecayTrajectoryFactory and ReferenceTrajectoryFactory 
/// with useAllFactories = false. In case the former
/// can't produce a trajectory from two given tracks, the tracks can still be used for 'ordinary'
/// reference trajectories (see also TrajectoryFactories.cff).


class CombinedTrajectoryFactory : public TrajectoryFactoryBase
{

public:

  CombinedTrajectoryFactory(const edm::ParameterSet &config);
  ~CombinedTrajectoryFactory() override;

  const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const reco::BeamSpot &beamSpot) const override;

  const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const ExternalPredictionCollection &external,
							   const reco::BeamSpot &beamSpot) const override;

  CombinedTrajectoryFactory* clone() const override { return new CombinedTrajectoryFactory(*this); }

private:

  std::vector<TrajectoryFactoryBase*> theFactories;
  bool                                theUseAllFactories; /// use not only the first 'successful'?
};

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

using namespace std;

CombinedTrajectoryFactory::CombinedTrajectoryFactory( const edm::ParameterSet & config ) :
  TrajectoryFactoryBase( config ), theUseAllFactories(config.getParameter<bool>("useAllFactories"))
{
  vector<string> factoryNames = config.getParameter< vector<string> >( "TrajectoryFactoryNames" );
  vector<string>::iterator itFactoryName;
  for ( itFactoryName = factoryNames.begin(); itFactoryName != factoryNames.end(); ++itFactoryName )
  {
    // auto_ptr to avoid missing a delete due to throw...
    std::unique_ptr<TObjArray> namePset(TString((*itFactoryName).c_str()).Tokenize(","));
    if (namePset->GetEntriesFast() != 2) {
      throw cms::Exception("BadConfig") << "@SUB=CombinedTrajectoryFactory"
                                        << "TrajectoryFactoryNames must contain 2 comma "
                                        << "separated strings, but is '" << *itFactoryName << "'";
    }
    const edm::ParameterSet factoryCfg 
      = config.getParameter<edm::ParameterSet>(namePset->At(1)->GetName());
    theFactories.push_back(TrajectoryFactoryPlugin::get()->create(namePset->At(0)->GetName(),
								  factoryCfg));
  }
}

 
CombinedTrajectoryFactory::~CombinedTrajectoryFactory( void ) {}


const CombinedTrajectoryFactory::ReferenceTrajectoryCollection
CombinedTrajectoryFactory::trajectories(const edm::EventSetup &setup,
					const ConstTrajTrackPairCollection &tracks,
					const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;
  ReferenceTrajectoryCollection tmpTrajectories; // outside loop for efficiency

  vector< TrajectoryFactoryBase* >::const_iterator itFactory;
  for ( itFactory = theFactories.begin(); itFactory != theFactories.end(); ++itFactory )
  {
    tmpTrajectories = ( *itFactory )->trajectories(setup, tracks, beamSpot);
    trajectories.insert(trajectories.end(), tmpTrajectories.begin(), tmpTrajectories.end());

    if (!theUseAllFactories && !trajectories.empty()) break;
  }

  return trajectories;
}

const CombinedTrajectoryFactory::ReferenceTrajectoryCollection
CombinedTrajectoryFactory::trajectories(const edm::EventSetup &setup,
					const ConstTrajTrackPairCollection &tracks,
					const ExternalPredictionCollection &external,
					const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;
  ReferenceTrajectoryCollection tmpTrajectories; // outside loop for efficiency

  vector< TrajectoryFactoryBase* >::const_iterator itFactory;
  for ( itFactory = theFactories.begin(); itFactory != theFactories.end(); ++itFactory )
  {
    tmpTrajectories = ( *itFactory )->trajectories(setup, tracks, external, beamSpot);
    trajectories.insert(trajectories.end(), tmpTrajectories.begin(), tmpTrajectories.end());

    if (!theUseAllFactories && !trajectories.empty()) break;
  }

  return trajectories;
}


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, CombinedTrajectoryFactory, "CombinedTrajectoryFactory" );
