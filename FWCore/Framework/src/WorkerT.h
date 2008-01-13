#ifndef FWCore_Framework_WorkerT_h
#define FWCore_Framework_WorkerT_h

/*----------------------------------------------------------------------
  
WorkerT: Code common to all workers.

$Id: WorkerT.h,v 1.22 2007/09/18 18:06:47 chrjones Exp $

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
    WorkerT(std::auto_ptr<T>,
		   ModuleDescription const&,
		   WorkerParams const&);

    virtual ~WorkerT();


  template <typename ModType>
  static std::auto_ptr<T> makeOne(ModuleDescription const& md,
					   WorkerParams const& wp) {
    std::auto_ptr<ModType> module = std::auto_ptr<ModType>(new ModType(*wp.pset_));
    module->setModuleDescription(md);
    return std::auto_ptr<T>(module.release());
  }

  protected:
    T & module() {return *module_;}
    T const& module() const {return *module_;}
    boost::shared_ptr<T> moduleSharedPtr() const {return module_;}

  private:
    virtual void implBeginJob(EventSetup const&) ;
    virtual void implEndJob() ;
    virtual void implRespondToOpenInputFile(FileBlock const& fb);
    virtual void implRespondToCloseInputFile(FileBlock const& fb);
    virtual void implRespondToOpenOutputFiles(FileBlock const& fb);
    virtual void implRespondToCloseOutputFiles(FileBlock const& fb);

    boost::shared_ptr<T> module_;
  };

  template <typename T>
  inline
  WorkerT<T>::WorkerT(std::auto_ptr<T> ed,
		 ModuleDescription const& md,
		 WorkerParams const& wp) :
    Worker(md, wp),
    module_(ed) {
  }

  template <typename T>
  WorkerT<T>::~WorkerT() {
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
