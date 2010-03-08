// Do not include .h from plugin directory, but locally:
#include "CombinedTrajectoryFactory.h"

#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include <TString.h>
#include <TObjArray.h>

using namespace std;


CombinedTrajectoryFactory::CombinedTrajectoryFactory( const edm::ParameterSet & config ) :
  TrajectoryFactoryBase( config ), theUseAllFactories(config.getParameter<bool>("useAllFactories"))
{
  vector<string> factoryNames = config.getParameter< vector<string> >( "TrajectoryFactoryNames" );
  vector<string>::iterator itFactoryName;
  for ( itFactoryName = factoryNames.begin(); itFactoryName != factoryNames.end(); ++itFactoryName )
  {
    // auto_ptr to avoid missing a delete due to throw...
    std::auto_ptr<TObjArray> namePset(TString((*itFactoryName).c_str()).Tokenize(","));
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
CombinedTrajectoryFactory::trajectories( const edm::EventSetup & setup,
					 const ConstTrajTrackPairCollection & tracks ) const
{
  ReferenceTrajectoryCollection trajectories;
  ReferenceTrajectoryCollection tmpTrajectories; // outside loop for efficiency

  vector< TrajectoryFactoryBase* >::const_iterator itFactory;
  for ( itFactory = theFactories.begin(); itFactory != theFactories.end(); ++itFactory )
  {
    tmpTrajectories = ( *itFactory )->trajectories( setup, tracks );
    trajectories.insert(trajectories.end(), tmpTrajectories.begin(), tmpTrajectories.end());

    if (!theUseAllFactories && !trajectories.empty()) break;
  }

  return trajectories;
}

const CombinedTrajectoryFactory::ReferenceTrajectoryCollection
CombinedTrajectoryFactory::trajectories( const edm::EventSetup & setup,
					 const ConstTrajTrackPairCollection& tracks,
					 const ExternalPredictionCollection& external ) const
{
  ReferenceTrajectoryCollection trajectories;
  ReferenceTrajectoryCollection tmpTrajectories; // outside loop for efficiency

  vector< TrajectoryFactoryBase* >::const_iterator itFactory;
  for ( itFactory = theFactories.begin(); itFactory != theFactories.end(); ++itFactory )
  {
    tmpTrajectories = ( *itFactory )->trajectories( setup, tracks, external );
    trajectories.insert(trajectories.end(), tmpTrajectories.begin(), tmpTrajectories.end());

    if (!theUseAllFactories && !trajectories.empty()) break;
  }

  return trajectories;
}


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, CombinedTrajectoryFactory, "CombinedTrajectoryFactory" );
