#ifndef PhysicsTools_UtilAlgos_interface_EDAnalyzerWrapper_h
#define PhysicsTools_UtilAlgos_interface_EDAnalyzerWrapper_h

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


/**
   \class    EDAnalyzerWrapper EDAnalyzerWrapper.h "PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h"
   \brief    Wrapper class around a class of type BasicAnalyzer to "convert" it into a full EDAnalyzer

   This template class is a wrapper round classes of type BasicAnalyzer as defined in in the
   BasicAnalyzer.h file of this package. From this class the wrapper expects the following
   member functions:

   + a contructor with a const edm::ParameterSet& and a TFileDirectory& as input.
   + a beginJob function
   + a endJob function
   + a analyze function with an const edm::EventBase& as input

   these functions are called within the wrapper. The wrapper translates the common class into
   a basic EDAnalyzer as shown below:

   #include "PhysicsTools/PatExamples/interface/BasicMuonAnalyzer.h"
   #include "PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h"

   typedef edm::AnalyzerWrapper<BasicMuonAnalyzer> WrappedEDAnalyzer;

   #include "FWCore/Framework/interface/MakerMacros.h"
   DEFINE_FWK_MODULE(WrappedEDAnalyzer);

   With this wrapper class we have the use case in mind that you keep classes, which easily can
   be used both within the full framework and within FWLite.

   NOTE: in the current implementation this wrapper class does not support use of the EventSetup.
   If you want to make use of this feature we recommend you to start from an EDAnalyzer from the
   very beginning and just to stay within the full framework.
*/


namespace edm {

  template<class T>
  class AnalyzerWrapper : public EDAnalyzer {

  public:
    /// default contructor
    AnalyzerWrapper(const edm::ParameterSet& cfg);
    /// default destructor
    virtual ~AnalyzerWrapper(){};
    /// everything which has to be done before the event loop
    virtual void beginJob() { analyzer_->beginJob(); }
    /// everything which has to be done during the event loop. NOTE: We can't use the eventSetup in FWLite so ignore it
    virtual void analyze(edm::Event const & event, const edm::EventSetup& eventSetup){ analyzer_->analyze(event); }
    /// everything which has to be done after the event loop
    virtual void endJob() { analyzer_->endJob(); }

  protected:
    /// shared pointer to analysis class of type BasicAnalyzer
    boost::shared_ptr<T> analyzer_;
  };

  /// default contructor
  template<class T>
  AnalyzerWrapper<T>::AnalyzerWrapper(const edm::ParameterSet& cfg){
    // defined TFileService
    edm::Service<TFileService> fileService;
    // create analysis class of type BasicAnalyzer
    analyzer_ = boost::shared_ptr<T>( new T( cfg, fileService->tFileDirectory(), consumesCollector()) );
  }

}

#endif
