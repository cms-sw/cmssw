#ifndef MuonIsolation_MuIsoExtractorFactory_H
#define MuonIsolation_MuIsoExtractorFactory_H


namespace edm {class ParameterSet;}
namespace muonisolation { class MuIsoExtractor; }
#include "FWCore/PluginManager/interface/PluginFactory.h"

//class MuIsoExtractorFactory : public 
//    edmplugin::PluginFactory< muonisolation::MuIsoExtractor* (const edm::ParameterSet&) > { 
//public:
//  MuIsoExtractorFactory();
//  virtual ~MuIsoExtractorFactory();
//  static MuIsoExtractorFactory * get();
//};
typedef edmplugin::PluginFactory< muonisolation::MuIsoExtractor* (const edm::ParameterSet&) > MuIsoExtractorFactory;
#endif
