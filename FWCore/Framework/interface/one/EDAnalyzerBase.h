#ifndef FWCore_Framework_one_EDAnalyzerBase_h
#define FWCore_Framework_one_EDAnalyzerBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     one::EDAnalyzerBase
// 
/**\class one::EDAnalyzerBase EDAnalyzerBase.h "FWCore/Framework/interface/one/EDAnalyzerBase.h"

 Description: Base class for edm::one::EDAnalyzer<>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 21:21:21 GMT
//

// system include files
#include <mutex>

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

// forward declarations
namespace edm {

  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;

  namespace maker {
    template<typename T> class ModuleHolderT;
  }

  namespace one {

    class EDAnalyzerBase : public EDConsumerBase
    {
      
    public:
      template <typename T> friend class edm::WorkerT;
      template <typename T> friend class edm::maker::ModuleHolderT;
      
      typedef EDAnalyzerBase ModuleType;

      
      EDAnalyzerBase();
      virtual ~EDAnalyzerBase();
      
      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();

      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const { return moduleDescription_; }

    protected:
      
      void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func);

    private:
      bool doEvent(EventPrincipal& ep, EventSetup const& c,
                   ActivityRegistry*,
                   ModuleCallingContext const*);
      void doPreallocate(PreallocationConfiguration const&) {}
      void doBeginJob();
      void doEndJob();
      
      void doBeginRun(RunPrincipal& rp, EventSetup const& c,
                      ModuleCallingContext const*);
      void doEndRun(RunPrincipal& rp, EventSetup const& c,
                    ModuleCallingContext const*);
      void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                  ModuleCallingContext const*);
      void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                ModuleCallingContext const*);
      
      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doPreForkReleaseResources();
      void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

      
      void registerProductsAndCallbacks(EDAnalyzerBase const* module, ProductRegistry* reg);
      std::string workerType() const {return "WorkerT<EDAnalyzer>";}
      
      virtual void analyze(Event const&, EventSetup const&) = 0;
      virtual void beginJob() {}
      virtual void endJob(){}

      virtual void doBeginRun_(Run const& rp, EventSetup const& c);
      virtual void doEndRun_(Run const& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);
      virtual void doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);

      virtual SharedResourcesAcquirer createAcquirer();

      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      ModuleDescription moduleDescription_;
      std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
      
      SharedResourcesAcquirer resourcesAcquirer_;
      std::mutex mutex_;
    };
  }
}
#endif
