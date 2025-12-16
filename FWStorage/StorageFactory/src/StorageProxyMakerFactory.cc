#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWStorage/StorageFactory/interface/StorageProxyMakerFactory.h"

using namespace edm::storage;

EDM_REGISTER_VALIDATED_PLUGINFACTORY(StorageProxyMakerFactory, "CMS Storage Proxy Maker");
