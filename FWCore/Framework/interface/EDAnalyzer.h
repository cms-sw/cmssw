#ifndef FWCore_Framework_EDAnalyzer_h
#define FWCore_Framework_EDAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include <string>

// EDAnalyzer is the base class for all analyzer "modules".

namespace edm {

  class ModuleCallingContext;

  class EDAnalyzer : public EDConsumerBase {
  public:
    template <typename T> friend class WorkerT;
    typedef EDAnalyzer ModuleType;
    typedef WorkerT<EDAnalyzer> WorkerType;

    EDAnalyzer() : moduleDescription_(), current_context_(nullptr) {}
    virtual ~EDAnalyzer();
    
    std::string workerType() const {return "WorkerT<EDAnalyzer>";}

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();
    static   void prevalidate(ConfigurationDescriptions& );

    // Warning: the returned moduleDescription will be invalid during construction
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('analyze').
    CurrentProcessingContext const* currentContext() const;

    void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func);

  private:
    bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                 CurrentProcessingContext const* cpc,
                 ModuleCallingContext const* mcc);
    void doBeginJob();
    void doEndJob();
    bool doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                    CurrentProcessingContext const* cpc,
                    ModuleCallingContext const* mcc);
    bool doEndRun(RunPrincipal const& rp, EventSetup const& c,
                  CurrentProcessingContext const* cpc,
                  ModuleCallingContext const* mcc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                CurrentProcessingContext const* cpc,
                                ModuleCallingContext const* mcc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                              CurrentProcessingContext const* cpc,
                              ModuleCallingContext const* mcc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doPreForkReleaseResources();
    void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);
    void registerProductsAndCallbacks(EDAnalyzer const*, ProductRegistry* reg);

    virtual void analyze(Event const&, EventSetup const&) = 0;
    virtual void beginJob(){}
    virtual void endJob(){}
    virtual void beginRun(Run const&, EventSetup const&){}
    virtual void endRun(Run const&, EventSetup const&){}
    virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
    virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}
    virtual void preForkReleaseResources() {}
    virtual void postForkReacquireResources(unsigned int /*iChildIndex*/, unsigned int /*iNumberOfChildren*/) {}

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }
    ModuleDescription moduleDescription_;

    CurrentProcessingContext const* current_context_;

    std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
  };
}

#endif
