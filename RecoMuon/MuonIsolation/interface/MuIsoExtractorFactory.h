#ifndef MuonIsolation_MuIsoExtractorFactory_H
#define MuonIsolation_MuIsoExtractorFactory_H


namespace edm {class ParameterSet;}
namespace muonisolation { class MuIsoExtractor; }
#include <PluginManager/PluginFactory.h>

class MuIsoExtractorFactory : public 
    seal::PluginFactory< muonisolation::MuIsoExtractor* (const edm::ParameterSet&) > { 
public:
  MuIsoExtractorFactory();
  virtual ~MuIsoExtractorFactory();
  static MuIsoExtractorFactory * get();
};
#endif
