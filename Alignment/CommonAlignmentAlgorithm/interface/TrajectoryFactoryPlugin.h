#ifndef Alignment_CommonAlignmentAlgorithm_TrajectoryFactoryPlugin_h
#define Alignment_CommonAlignmentAlgorithm_TrajectoryFactoryPlugin_h

#include "PluginManager/PluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryBase.h"

#include <string>

/// A PluginFactory that produces factories that inherit from TrajectoryFactoryBase.

 
class TrajectoryFactoryPlugin : public seal::PluginFactory< TrajectoryFactoryBase *( const edm::ParameterSet & ) >
{

public:

  TrajectoryFactoryPlugin( void );

  static TrajectoryFactoryPlugin* get( void );

  static TrajectoryFactoryBase* getFactory( std::string factory, const edm::ParameterSet & config );

private:

  static TrajectoryFactoryPlugin theInstance;

};

#endif
