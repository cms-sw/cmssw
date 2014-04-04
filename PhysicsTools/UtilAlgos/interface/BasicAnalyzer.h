#ifndef PhysicsTools_UtilAlgos_interface_BasicAnalyzer_h
#define PhysicsTools_UtilAlgos_interface_BasicAnalyzer_h

#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

/**
   \class BasicAnalyzer BasicAnalyzer.h "PhysicsTools/UtilAlgos/interface/BasicAnalyzer.h"
   \brief Abstract base class for FWLite and EDM friendly analyzers

   Abstract base class for FWLite and EDM friendly analyzers. This class provides a proper
   interface needed for the EDAnalyzerWrapper and FWLiteAnalyzerWrapper template classes.
   Classes of type BasicAnalyzer can be wrapped into an EDAnalyzer as shown in the example
   below:

   #include "PhysicsTools/PatExamples/interface/BasicMuonAnalyzer.h"
   #include "PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h"

   typedef edm::AnalyzerWrapper<BasicMuonAnalyzer> WrappedEDAnalyzer;

   #include "FWCore/Framework/interface/MakerMacros.h"
   DEFINE_FWK_MODULE(WrappedEDAnalyzer);

   Alternatively they can be wrapped into a FWLiteAnalyzer which provides basic functionality
   of reading configuration files and event looping as shown in the example below:

   #include "PhysicsTools/PatExamples/interface/BasicMuonAnalyzer.h"
   #include "PhysicsTools/UtilAlgos/interface/FWLiteAnalyzerWrapper.h"

   typedef fwlite::AnalyzerWrapper<BasicMuonAnalyzer> WrappedFWLiteAnalyzer;
   ...

   In both examples BasicMuonAnalyzer is derived from the BasicAnalyzer class. For more
   information have a look into the class description of the corresponding wrapper classes.
*/


namespace edm {

  class BasicAnalyzer {
  public:
    /// default constructor
    BasicAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fileService){};
    BasicAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fileService, edm::ConsumesCollector&& iC){};
    /// default destructor
    virtual ~BasicAnalyzer(){};

    /**
       The following functions have to be implemented for any class
       derived from BasicAnalyzer; these functions are called in
       the EDAnalyzerWrapper class or in the FWLiteAnalyzerWrapper
       class.
    **/

    /// everything that needs to be done before the event loop
    virtual void beginJob()=0;
    /// everything that needs to be done after the event loop
    virtual void endJob()  =0;
    /// everything that needs to be done during the event loop
    virtual void analyze(const edm::EventBase& event)=0;
  };

}

#endif
