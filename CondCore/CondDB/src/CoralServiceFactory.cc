#include "CondCore/CondDB/interface/CoralServiceFactory.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CoralKernel/Service.h"

EDM_REGISTER_PLUGINFACTORY(cond::CoralServicePluginFactory, "CoralService");

cond::CoralServiceFactory::~CoralServiceFactory() {}

cond::CoralServiceFactory::CoralServiceFactory() {}

cond::CoralServiceFactory* cond::CoralServiceFactory::get() {
  static cond::CoralServiceFactory singleInstance_;
  return &singleInstance_;
}

coral::Service* cond::CoralServiceFactory::create(const std::string& componentname) const {
  std::unique_ptr<cond::CoralServiceWrapperBase> sp{CoralServicePluginFactory::get()->create(componentname)};
  return sp->create(componentname);
}
