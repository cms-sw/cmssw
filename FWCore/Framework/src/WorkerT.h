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
                               bool anyProductProduced) {
      module_->setEventSelectionInfo(outputModulePathPositions, anyProductProduced);
    }

  protected:
    T& module() {return *module_;}
    T const& module() const {return *module_;}

  private:
    virtual bool implDoBegin(EventPrincipal& ep, EventSetup const& c,
                            CurrentProcessingContext const* cpc);
    virtual bool implDoEnd(EventPrincipal& ep, EventSetup const& c,
                            CurrentProcessingContext const* cpc);
    virtual bool implDoBegin(RunPrincipal& rp, EventSetup const& c,
                            CurrentProcessingContext const* cpc);
    virtual bool implDoEnd(RunPrincipal& rp, EventSetup const& c,
                            CurrentProcessingContext const* cpc);
    virtual bool implDoBegin(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                            CurrentProcessingContext const* cpc);
    virtual bool implDoEnd(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                            CurrentProcessingContext const* cpc);
    virtual void implBeginJob() ;
    virtual void implEndJob() ;
    virtual void implRespondToOpenInputFile(FileBlock const& fb);
    virtual void implRespondToCloseInputFile(FileBlock const& fb);
    virtual void implRespondToOpenOutputFiles(FileBlock const& fb);
    virtual void implRespondToCloseOutputFiles(FileBlock const& fb);
    virtual void implPreForkReleaseResources();
    virtual void implPostForkReacquireResources(unsigned int iChildIndex, 
                                               unsigned int iNumberOfChildren);
     virtual std::string workerType() const;

    std::auto_ptr<T> module_;
  };

  template<typename T>
  inline
  WorkerT<T>::WorkerT(std::auto_ptr<T> ed, ModuleDescription const& md, WorkerParams const& wp) :
    Worker(md, wp),
    module_(ed) {
    assert(module_.get() != 0);
    module_->setModuleDescription(md);
    module_->registerProductsAndCallbacks(module_.get(), wp.reg_);
  }

  template<typename T>
  WorkerT<T>::~WorkerT() {
  }

  template<typename T>
  inline
  bool 
  WorkerT<T>::implDoBegin(EventPrincipal& ep, EventSetup const& c, CurrentProcessingContext const* cpc) {
    UnscheduledHandlerSentry s(getUnscheduledHandler(ep), cpc);
    boost::shared_ptr<Worker> sentry(this,[&ep](Worker* obj) {obj->postDoEvent(ep);});
    return module_->doEvent(ep, c, cpc);
  }

  template<typename T>
  inline
  bool 
  WorkerT<T>::implDoEnd(EventPrincipal&, EventSetup const&, CurrentProcessingContext const*) {
    return false;
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoBegin(RunPrincipal& rp, EventSetup const& c, CurrentProcessingContext const* cpc) {
    return module_->doBeginRun(rp, c, cpc);
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(RunPrincipal& rp, EventSetup const& c, CurrentProcessingContext const* cpc) {
    return module_->doEndRun(rp, c, cpc);
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoBegin(LuminosityBlockPrincipal& lbp, EventSetup const& c, CurrentProcessingContext const* cpc) {
    return module_->doBeginLuminosityBlock(lbp, c, cpc);
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(LuminosityBlockPrincipal& lbp, EventSetup const& c, CurrentProcessingContext const* cpc) {
    return module_->doEndLuminosityBlock(lbp, c, cpc);
  }

  template<typename T>
  inline
  std::string
  WorkerT<T>::workerType() const {
    return module_->workerType();
  }

  template<typename T>
  inline
  void
  WorkerT<T>::implBeginJob() {
    module_->doBeginJob();
  }

  template<typename T>
  inline
  void
  WorkerT<T>::implEndJob() {
    module_->doEndJob();
  }

  template<typename T>
  inline
  void
  WorkerT<T>::implRespondToOpenInputFile(FileBlock const& fb) {
    module_->doRespondToOpenInputFile(fb);
  }

  template<typename T>
  inline
  void
  WorkerT<T>::implRespondToCloseInputFile(FileBlock const& fb) {
    module_->doRespondToCloseInputFile(fb);
  }

  template<typename T>
  inline
  void
  WorkerT<T>::implRespondToOpenOutputFiles(FileBlock const& fb) {
    module_->doRespondToOpenOutputFiles(fb);
  }

  template<typename T>
  inline
  void
  WorkerT<T>::implRespondToCloseOutputFiles(FileBlock const& fb) {
    module_->doRespondToCloseOutputFiles(fb);
  }

  template<typename T>
  inline
  void 
  WorkerT<T>::implPreForkReleaseResources() {
    module_->doPreForkReleaseResources();
  }

  template<typename T>
  inline
  void 
  WorkerT<T>::implPostForkReacquireResources(unsigned int iChildIndex, 
                                            unsigned int iNumberOfChildren) {
    module_->doPostForkReacquireResources(iChildIndex, iNumberOfChildren);
  }  
}

#endif
