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

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

// forward declarations
namespace edm {

  class ModuleCallingContext;

  namespace one {

    class EDAnalyzerBase : public EDConsumerBase
    {
      
    public:
      template <typename T> friend class edm::WorkerT;
      typedef EDAnalyzerBase ModuleType;
      typedef WorkerT<EDAnalyzerBase> WorkerType;

      
      EDAnalyzerBase();
      virtual ~EDAnalyzerBase();
      
      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();

      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const { return moduleDescription_; }

    protected:
      // The returned pointer will be null unless the this is currently
      // executing its event loop function ('produce').
      CurrentProcessingContext const* currentContext() const;
      
      void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func);

    private:
      bool doEvent(EventPrincipal& ep, EventSetup const& c,
                   CurrentProcessingContext const* cpcp,
                   ModuleCallingContext const*);
      void doBeginJob();
      void doEndJob();
      
      void doBeginRun(RunPrincipal& rp, EventSetup const& c,
                      CurrentProcessingContext const* cpc,
                      ModuleCallingContext const*);
      void doEndRun(RunPrincipal& rp, EventSetup const& c,
                    CurrentProcessingContext const* cpc,
                    ModuleCallingContext const*);
      void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                  CurrentProcessingContext const* cpc,
                                  ModuleCallingContext const*);
      void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                CurrentProcessingContext const* cpc,
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

      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      ModuleDescription moduleDescription_;
      CurrentProcessingContext const* current_context_;
      std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;

    };
    
  }
}


#endif
