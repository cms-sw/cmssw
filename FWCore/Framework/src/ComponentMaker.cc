#include "FWCore/Framework/interface/ComponentMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  namespace eventsetup {

ComponentDescription 
ComponentMakerBaseHelper::createComponentDescription(ParameterSet const& iConfiguration,
                                           std::string const& iProcessName,
                                           ReleaseVersion const& iVersion,
                                           PassID const& iPass) const
{
  ComponentDescription description;
  description.type_  = iConfiguration.getParameter<std::string>("@module_type");
  description.label_ = iConfiguration.getParameter<std::string>("@module_label");

  description.releaseVersion_ = iVersion;
  description.pid_            = iConfiguration.id();
  description.processName_    = iProcessName;
  description.passID_         = iPass;
  return description;
}

} // namespace eventsetup
} // namespace edm
