#include "EventFilter/Utilities/interface/MicroStateService.h"


namespace evf{
  
  const edm::ModuleDescription MicroStateService::reservedMicroStateNames[mCOUNT] = 
    { edm::ModuleDescription("Dummy","Invalid"), edm::ModuleDescription("Dummy","FwkOvh"), 
      edm::ModuleDescription("Dummy","Idle"), edm::ModuleDescription("Dummy","Input"), 
      edm::ModuleDescription("Dummy","InputDone"), edm::ModuleDescription("Dummy","DQM"),
      edm::ModuleDescription("Dummy","EoL")};

  const std::string MicroStateService::default_return_="NotImplemented";

  MicroStateService::MicroStateService(const edm::ParameterSet& iPS, 
				       edm::ActivityRegistry& reg)  
  {
  }


  MicroStateService::~MicroStateService()
  {
  }



} //end namespace evf

