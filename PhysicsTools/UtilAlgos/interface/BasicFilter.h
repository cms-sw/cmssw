#ifndef PhysicsTools_UtilAlgos_interface_BasicFilter_h
#define PhysicsTools_UtilAlgos_interface_BasicFilter_h

#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

/**
   \class BasicFilter BasicFilter.h "PhysicsTools/UtilAlgos/interface/BasicFilter.h"
   \brief Abstract base class for FWLite and EDM friendly analyzers 
   
   Abstract base class for FWLite and EDM friendly analyzers. This class provides a proper 
   interface needed for the EDFilterWrapper and FWLiteFilterWrapper template classes. 
   Classes of type BasicFilter can be wrapped into an EDFilter as shown in the example 
   below:

   #include "PhysicsTools/PatExamples/interface/BasicMuonFilter.h"
   #include "PhysicsTools/UtilAlgos/interface/EDFilterWrapper.h"
   
   typedef edm::FilterWrapper<BasicMuonFilter> WrappedEDFilter;
   
   #include "FWCore/Framework/interface/MakerMacros.h"
   DEFINE_FWK_MODULE(WrappedEDFilter);   

   Alternatively they can be wrapped into a FWLiteFilter which provides basic functionality 
   of reading configuration files and event looping as shown in the example below: 

   #include "PhysicsTools/PatExamples/interface/BasicMuonFilter.h"
   #include "PhysicsTools/UtilAlgos/interface/FWLiteFilterWrapper.h"
   
   typedef fwlite::FilterWrapper<BasicMuonFilter> WrappedFWLiteFilter;
   ...

   In both examples BasicMuonFilter is derived from the BasicFilter class. For more 
   information have a look into the class description of the corresponding wrapper classes.
*/


namespace edm {

  class BasicFilter {
  public:
    /// default constructor
    BasicFilter(const edm::ParameterSet& cfg){};
    /// default destructor
    virtual ~BasicFilter(){};
    
    /**
       The following functions have to be implemented for any class
       derived from BasicFilter; these functions are called in the
       EDFilterWrapper class.
    **/
    
    /// everything that needs to be done during the event loop
    virtual bool filter(const edm::EventBase& event)=0;
    /// interface to filters as they exists in PhysicsTools/SelectorUtils
    vitrual bool operator ()(const edm::EventBase& event){return filter(event);}
  };
  
}

#endif
