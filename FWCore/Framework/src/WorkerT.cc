#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/OutputModule.h"

namespace edm{
  UnscheduledHandler* getUnscheduledHandler(EventPrincipal const& ep) {
    return ep.unscheduledHandler().get();
  }


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
    module_->doBeginRun(rp, c, cpc);
    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(RunPrincipal& rp, EventSetup const& c, CurrentProcessingContext const* cpc) {
    module_->doEndRun(rp, c, cpc);
    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoBegin(LuminosityBlockPrincipal& lbp, EventSetup const& c, CurrentProcessingContext const* cpc) {
    module_->doBeginLuminosityBlock(lbp, c, cpc);
    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(LuminosityBlockPrincipal& lbp, EventSetup const& c, CurrentProcessingContext const* cpc) {
    module_->doEndLuminosityBlock(lbp, c, cpc);
    return true;
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

  template<typename T>
  void WorkerT<T>::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                             bool anyProductProduced) {
    //do nothing for the regular case
  }

  
  template<>
  void WorkerT<OutputModule>::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                             bool anyProductProduced) {
    module_->setEventSelectionInfo(outputModulePathPositions, anyProductProduced);
  }

  template<>
  Worker::Types WorkerT<EDAnalyzer>::moduleType() const { return Worker::kAnalyzer;}
  template<>
  Worker::Types WorkerT<EDProducer>::moduleType() const { return Worker::kProducer;}
  template<>
  Worker::Types WorkerT<EDFilter>::moduleType() const { return Worker::kFilter;}
  template<>
  Worker::Types WorkerT<OutputModule>::moduleType() const { return Worker::kOutputModule;}

  
  //Explicitly instantiate our needed templates to avoid having the compiler
  // instantiate them in all of our libraries
  template class WorkerT<EDProducer>;
  template class WorkerT<EDFilter>;
  template class WorkerT<EDAnalyzer>;
  template class WorkerT<OutputModule>;
}
