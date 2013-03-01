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
    WorkerT(std::auto_ptr<T>,
            ModuleDescription const&,
            WorkerParams const&);

    virtual ~WorkerT();

  template<typename ModType>
  static std::auto_ptr<T> makeModule(ModuleDescription const&,
                                     ParameterSet const& pset) {
    std::auto_ptr<ModType> module = std::auto_ptr<ModType>(new ModType(pset));
    return std::auto_ptr<T>(module.release());
  }

  void setModule( std::auto_ptr<T>& iModule) {
     module_ = iModule;
     module_->setModuleDescription(description());
     
  }

    void setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                               bool anyProductProduced);
    
    virtual Types moduleType() const override;


  protected:
    T& module() {return *module_;}
    T const& module() const {return *module_;}

  private:
    virtual bool implDoBegin(EventPrincipal& ep, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual bool implDoEnd(EventPrincipal& ep, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual bool implDoBegin(RunPrincipal& rp, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual bool implDoEnd(RunPrincipal& rp, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual bool implDoBegin(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual bool implDoEnd(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                            CurrentProcessingContext const* cpc) override;
    virtual void implBeginJob() override;
    virtual void implEndJob() override;
    virtual void implRespondToOpenInputFile(FileBlock const& fb) override;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) override;
    virtual void implRespondToOpenOutputFiles(FileBlock const& fb) override;
    virtual void implRespondToCloseOutputFiles(FileBlock const& fb) override;
    virtual void implPreForkReleaseResources() override;
    virtual void implPostForkReacquireResources(unsigned int iChildIndex, 
                                               unsigned int iNumberOfChildren) override;
     virtual std::string workerType() const override;

    std::auto_ptr<T> module_;
  };

}

#endif
