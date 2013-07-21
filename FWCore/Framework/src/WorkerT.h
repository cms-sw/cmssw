#ifndef FWCore_Framework_WorkerT_h
#define FWCore_Framework_WorkerT_h

/*----------------------------------------------------------------------

WorkerT: Code common to all workers.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerParams.h"

#include <memory>

namespace edm {
  UnscheduledHandler* getUnscheduledHandler(EventPrincipal const& ep);

  template<typename T>
  class WorkerT : public Worker {
  public:
    typedef T ModuleType;
    typedef WorkerT<T> WorkerType;
    WorkerT(std::unique_ptr<T>&&,
            ModuleDescription const&,
            WorkerParams const&);

    virtual ~WorkerT();

  template<typename ModType>
  static std::unique_ptr<T> makeModule(ModuleDescription const&,
                                     ParameterSet const& pset) {
    std::unique_ptr<ModType> module = std::unique_ptr<ModType>(new ModType(pset));
    return std::unique_ptr<T>(module.release());
  }

  void setModule( std::unique_ptr<T>&& iModule) {
    module_ = std::move(iModule);
     module_->setModuleDescription(description());
     
  }
    
    virtual Types moduleType() const override;

    virtual void updateLookup(BranchType iBranchType,
                              ProductHolderIndexHelper const&) override;


  protected:
    T& module() {return *module_;}
    T const& module() const {return *module_;}

  private:
    virtual bool implDo(EventPrincipal& ep, EventSetup const& c,
                        CurrentProcessingContext const* cpc) override;
    virtual bool implDoBegin(RunPrincipal& rp, EventSetup const& c,
                             CurrentProcessingContext const* cpc) override;
    virtual bool implDoStreamBegin(StreamID id, RunPrincipal& rp, EventSetup const& c,
                                   CurrentProcessingContext const* cpc) override;
    virtual bool implDoStreamEnd(StreamID id, RunPrincipal& rp, EventSetup const& c,
                                 CurrentProcessingContext const* cpc) override;
    virtual bool implDoEnd(RunPrincipal& rp, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual bool implDoBegin(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual bool implDoStreamBegin(StreamID id, LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                   CurrentProcessingContext const* cpc) override;
    virtual bool implDoStreamEnd(StreamID id, LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                 CurrentProcessingContext const* cpc) override;
    virtual bool implDoEnd(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                           CurrentProcessingContext const* cpc) override;
    virtual void implBeginJob() override;
    virtual void implEndJob() override;
    virtual void implBeginStream(StreamID) override;
    virtual void implEndStream(StreamID) override;
    virtual void implRespondToOpenInputFile(FileBlock const& fb) override;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) override;
    virtual void implRespondToOpenOutputFiles(FileBlock const& fb) override;
    virtual void implRespondToCloseOutputFiles(FileBlock const& fb) override;
    virtual void implPreForkReleaseResources() override;
    virtual void implPostForkReacquireResources(unsigned int iChildIndex, 
                                               unsigned int iNumberOfChildren) override;
     virtual std::string workerType() const override;

    std::unique_ptr<T> module_;
  };

}

#endif
