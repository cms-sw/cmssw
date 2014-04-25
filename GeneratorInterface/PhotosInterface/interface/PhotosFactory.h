#ifndef GeneratorInterface_PhotosInterface_PhotosFactory_H
#define GeneratorInterface_PhotosInterface_PhotosFactory_H
 
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"

typedef edmplugin::PluginFactory<gen::PhotosInterfaceBase* (edm::ParameterSet const&)>  PhotosFactory;

#endif
