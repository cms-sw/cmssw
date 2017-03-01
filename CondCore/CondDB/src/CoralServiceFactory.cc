#include "CondCore/CondDB/interface/CoralServiceFactory.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CoralKernel/Service.h"

EDM_REGISTER_PLUGINFACTORY(cond::CoralServicePluginFactory,"CoralService");


cond::CoralServiceFactory::~CoralServiceFactory() {
}

cond::CoralServiceFactory::CoralServiceFactory() {
}


cond::CoralServiceFactory* 
cond::CoralServiceFactory::get() {
  static cond::CoralServiceFactory singleInstance_;
  return &singleInstance_;
}

coral::Service*
cond::CoralServiceFactory::create(const std::string& componentname) const {
 coral::Service* sp=CoralServicePluginFactory::get()->create(componentname,componentname);
 if(sp==0) {
   throw cond::Exception("CoralServiceFactory")
     << "CoralServiceFactory:\n"
     << "Cannot find coral service: "
     << componentname << "\n"
     << "Perhaps the name is misspelled or is not a Plugin?\n"
     << "Try running EdmPluginDump to obtain a list of available Plugins.";
 }
 return sp;
}
