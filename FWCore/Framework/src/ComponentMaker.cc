#include "FWCore/Framework/interface/ComponentMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edm {
  namespace eventsetup {

ComponentDescription 
ComponentMakerBaseHelper::createComponentDescription(ParameterSet const& iConfiguration) const
{
  ComponentDescription description;
  description.type_  = iConfiguration.getParameter<std::string>("@module_type");
  description.label_ = iConfiguration.getParameter<std::string>("@module_label");

  description.pid_            = iConfiguration.id();
  return description;
}

} // namespace eventsetup
} // namespace edm
