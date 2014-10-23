#ifndef FWCore_Framework_EDAnalyzer_h
#define FWCore_Framework_EDAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

#include <string>
#include <mutex>

// EDAnalyzer is the base class for all analyzer "modules".

namespace edm {

  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;
  class ProductRegistry;
  class ThinnedAssociationsHelper;

  namespace maker {
    template<typename T> class ModuleHolderT;
  }

  class EDAnalyzer : public EDConsumerBase {
  public:
    template <typename T> friend class maker::ModuleHolderT;
    template <typename T> friend class WorkerT;
    typedef EDAnalyzer ModuleType;

    EDAnalyzer();
    virtual ~EDAnalyzer();
    
    std::string workerType() const {return "WorkerT<EDAnalyzer>";}

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();
    static   void prevalidate(ConfigurationDescriptions& );

    // Warning: the returned moduleDescription will be invalid during construction
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }

    void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func);

  private:
    bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                 ActivityRegistry* act,
                 ModuleCallingContext const* mcc);
    void doPreallocate(PreallocationConfiguration const&) {}
    void doBeginJob();
    void doEndJob();
    bool doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                    ModuleCallingContext const* mcc);
    bool doEndRun(RunPrincipal const& rp, EventSetup const& c,
                  ModuleCallingContext const* mcc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                ModuleCallingContext const* mcc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                              ModuleCallingContext const* mcc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doPreForkReleaseResources();
    void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);
    void doRegisterThinnedAssociations(ProductRegistry const&,
                                       ThinnedAssociationsHelper&) { }

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
    SharedResourcesAcquirer resourceAcquirer_;
    std::mutex mutex_;

    std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
  };
}

#endif
