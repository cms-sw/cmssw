#include "FWCore/Framework/interface/ComponentMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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


void
ComponentMakerBaseHelper::logInfoWhenSharing(ParameterSet const& iConfiguration) const {

   std::string edmtype = iConfiguration.getParameter<std::string>("@module_edm_type");
   std::string modtype = iConfiguration.getParameter<std::string>("@module_type");
   std::string label = iConfiguration.getParameter<std::string>("@module_label");
   edm::LogInfo("EventSetupSharing") << "Sharing " << edmtype << ": class=" << modtype << " label='" << label << "'";
}

} // namespace eventsetup
} // namespace edm
