#include "EventFilter/Utilities/interface/MicroStateService.h"


namespace evf{
  
  const edm::ModuleDescription MicroStateService::reservedMicroStateNames[mCOUNT] = 
    { 
      edm::ModuleDescription("Dummy","Invalid"),
      edm::ModuleDescription("Dummy","Idle"),
      edm::ModuleDescription("Dummy","FwkOvhSrc"), 
      edm::ModuleDescription("Dummy","FwkOvhMod"), 
      edm::ModuleDescription("Dummy","FwkEoL"), 
      edm::ModuleDescription("Dummy","Input"), 
      edm::ModuleDescription("Dummy","DQM"),
      edm::ModuleDescription("Dummy","BoL"), 
      edm::ModuleDescription("Dummy","EoL"),
      edm::ModuleDescription("Dummy","GlobalEoL")};

  const std::string MicroStateService::default_return_="NotImplemented";

  MicroStateService::MicroStateService(const edm::ParameterSet& iPS, 
				       edm::ActivityRegistry& reg)  
  {
  }


  MicroStateService::~MicroStateService()
  {
  }



} //end namespace evf

