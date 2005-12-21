#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "EventFilter/Utilities/interface/ModuleWeb.h"

using namespace evf;

ModuleWebRegistry::ModuleWebRegistry(const edm::ParameterSet &ps){
}



bool ModuleWebRegistry::checkWeb(const std::string &name){return clm_.find(name) != clm_.end();}

void ModuleWebRegistry::invoke(xgi::Input *in, xgi::Output *out, const std::string &name)
{
  idct i = clm_.find(name);
  if(i != clm_.end())
    (*i).second->defaultWebPage(in,out);

}

void ModuleWebRegistry::clear(){clm_.clear();}


#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
using namespace edm::serviceregistry;

typedef ParameterSetMaker<ModuleWebRegistry> maker;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE_MAKER(ModuleWebRegistry,maker);
