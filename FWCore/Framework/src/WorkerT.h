#ifndef FWCore_Framework_WorkerT_h
#define FWCore_Framework_WorkerT_h

/*----------------------------------------------------------------------
  
WorkerT: Code common to all workers.

$Id: WorkerT.h,v 1.1 2008/01/13 01:12:35 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerParams.h"

namespace edm {

  template <typename T>
  class WorkerT : public Worker {
  public:
    typedef T ModuleType;
    typedef WorkerT<T> WorkerType;
    WorkerT(std::auto_ptr<T>,
		   ModuleDescription const&,
		   WorkerParams const&);

    virtual ~WorkerT();


  template <typename ModType>
  static std::auto_ptr<T> makeModule(ModuleDescription const& md,
					   ParameterSet const& pset) {
    std::auto_ptr<ModType> module = std::auto_ptr<ModType>(new ModType(pset));
    return std::auto_ptr<T>(module.release());
  }


  protected:
    T & module() {return *module_;}
    T const& module() const {return *module_;}

  private:
    virtual bool implDoWork(EventPrincipal& ep, EventSetup const& c,
                            BranchActionType,
                            CurrentProcessingContext const* cpc);
    virtual bool implDoWork(RunPrincipal& rp, EventSetup const& c,
                            BranchActionType bat,
                            CurrentProcessingContext const* cpc);
    virtual bool implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                            BranchActionType bat,
                            CurrentProcessingContext const* cpc);
    virtual void implBeginJob(EventSetup const&) ;
    virtual void implEndJob() ;
    virtual void implRespondToOpenInputFile(FileBlock const& fb);
    virtual void implRespondToCloseInputFile(FileBlock const& fb);
    virtual void implRespondToOpenOutputFiles(FileBlock const& fb);
    virtual void implRespondToCloseOutputFiles(FileBlock const& fb);
    virtual std::string workerType() const;

    boost::shared_ptr<T> module_;
  };

  template <typename T>
  inline
  WorkerT<T>::WorkerT(std::auto_ptr<T> ed,
		 ModuleDescription const& md,
		 WorkerParams const& wp) :
    Worker(md, wp),
    module_(ed) {
    module_->setModuleDescription(md);
    module_->registerAnyProducts(module_, wp.reg_);
  }

  template <typename T>
  WorkerT<T>::~WorkerT() {
  }


  template <typename T>
  bool 
  WorkerT<T>::implDoWork(EventPrincipal& ep, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    return module_->doEvent(ep, c, cpc);
  }

  template <typename T>
  bool
  WorkerT<T>::implDoWork(RunPrincipal& rp, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    return (bat == BranchActionBegin ?
	module_->doBeginRun(rp, c, cpc) :
	module_->doEndRun(rp, c, cpc));
  }

  template <typename T>
  bool
  WorkerT<T>::implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			   BranchActionType bat,
			   CurrentProcessingContext const* cpc) {
    return (bat == BranchActionBegin ?
	module_->doBeginLuminosityBlock(lbp, c, cpc) :
	module_->doEndLuminosityBlock(lbp, c, cpc));
  }

  template <typename T>
  std::string
  WorkerT<T>::workerType() const {
    return module_->workerType();
  }
  
  template <typename T>
  void
  WorkerT<T>::implBeginJob(EventSetup const& es) {
    module_->doBeginJob(es);
  }

  template <typename T>
  void
  WorkerT<T>::implEndJob() {
    module_->doEndJob();
  }
  
  template <typename T>
  void
  WorkerT<T>::implRespondToOpenInputFile(FileBlock const& fb) {
    module_->doRespondToOpenInputFile(fb);
  }

  template <typename T>
  void
  WorkerT<T>::implRespondToCloseInputFile(FileBlock const& fb) {
    module_->doRespondToCloseInputFile(fb);
  }

  template <typename T>
  void
  WorkerT<T>::implRespondToOpenOutputFiles(FileBlock const& fb) {
    module_->doRespondToOpenOutputFiles(fb);
  }

  template <typename T>
  void
  WorkerT<T>::implRespondToCloseOutputFiles(FileBlock const& fb) {
    module_->doRespondToCloseOutputFiles(fb);
  }
}

#endif
