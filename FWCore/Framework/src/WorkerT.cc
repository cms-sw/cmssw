#include "boost/mpl/if.hpp"

#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/one/EDFilterBase.h"
#include "FWCore/Framework/interface/one/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "FWCore/Framework/interface/global/EDProducerBase.h"
#include "FWCore/Framework/interface/global/EDFilterBase.h"
#include "FWCore/Framework/interface/global/EDAnalyzerBase.h"

#include "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDFilterAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"

namespace edm{
  namespace workerimpl {
    template<typename T>
    struct has_stream_functions {
      static bool constexpr value = false;
    };

    template<>
    struct has_stream_functions<edm::global::EDProducerBase> {
      static bool constexpr value = true;
    };
    
    template<>
    struct has_stream_functions<edm::global::EDFilterBase> {
      static bool constexpr value = true;
    };
    
    template<>
    struct has_stream_functions<edm::global::EDAnalyzerBase> {
      static bool constexpr value = true;
    };
    
    template<>
    struct has_stream_functions<edm::stream::EDProducerAdaptorBase> {
      static bool constexpr value = true;
    };
    
    template<>
    struct has_stream_functions<edm::stream::EDFilterAdaptorBase> {
      static bool constexpr value = true;
    };

    template<>
    struct has_stream_functions<edm::stream::EDAnalyzerAdaptorBase> {
      static bool constexpr value = true;
    };

    struct DoNothing {
      template< typename... T>
      inline void operator()(const T&...) {}
    };
    
    template<typename T>
    struct DoBeginStream {
      inline void operator()(WorkerT<T>* iWorker, StreamID id) {
        iWorker->callWorkerBeginStream(0,id);
      }
    };

    template<typename T>
    struct DoEndStream {
      inline void operator()(WorkerT<T>* iWorker, StreamID id) {
        iWorker->callWorkerEndStream(0,id);
      }
    };

    template<typename T, typename P>
    struct DoStreamBeginTrans {
      inline void operator() (WorkerT<T>* iWorker, StreamID id, P& rp,
                              EventSetup const& c,
                              ModuleCallingContext const* mcc) {
        iWorker->callWorkerStreamBegin(0,id,rp,c, mcc);
      }
    };

    template<typename T, typename P>
    struct DoStreamEndTrans {
      inline void operator() (WorkerT<T>* iWorker, StreamID id, P& rp,
                              EventSetup const& c,
                              ModuleCallingContext const* mcc) {
        iWorker->callWorkerStreamEnd(0,id,rp,c, mcc);
      }
    };
  }
  
  UnscheduledHandler* getUnscheduledHandler(EventPrincipal const& ep) {
    return ep.unscheduledHandler().get();
  }


  template<typename T>
  inline
  WorkerT<T>::WorkerT(std::shared_ptr<T> ed, ModuleDescription const& md, ExceptionToActionTable const* actions) :
  Worker(md, actions),
  module_(ed) {
    assert(module_ != 0);
  }
  
  template<typename T>
  WorkerT<T>::~WorkerT() {
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDo(EventPrincipal& ep, EventSetup const& c, ModuleCallingContext const* mcc) {
    std::shared_ptr<Worker> sentry(this,[&ep](Worker* obj) {obj->postDoEvent(ep);});
    return module_->doEvent(ep, c, activityRegistry(), mcc);
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoBegin(RunPrincipal& rp, EventSetup const& c, ModuleCallingContext const* mcc) {
    module_->doBeginRun(rp, c, mcc);
    return true;
  }
  
  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamBegin(D, StreamID id, RunPrincipal& rp,
                                    EventSetup const& c,
                                    ModuleCallingContext const* mcc) {
    module_->doStreamBeginRun(id, rp, c, mcc);
  }

  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamEnd(D, StreamID id, RunPrincipal& rp,
                                    EventSetup const& c,
                                    ModuleCallingContext const* mcc) {
    module_->doStreamEndRun(id, rp, c, mcc);
  }

  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoStreamBegin(StreamID id, RunPrincipal& rp, EventSetup const& c,
                                ModuleCallingContext const* mcc) {
    typename boost::mpl::if_c<workerimpl::has_stream_functions<T>::value,
    workerimpl::DoStreamBeginTrans<T,RunPrincipal>,
    workerimpl::DoNothing>::type might_call;
    might_call(this,id,rp,c, mcc);
    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoStreamEnd(StreamID id, RunPrincipal& rp, EventSetup const& c,
                              ModuleCallingContext const* mcc) {
    typename boost::mpl::if_c<workerimpl::has_stream_functions<T>::value,
    workerimpl::DoStreamEndTrans<T,RunPrincipal>,
    workerimpl::DoNothing>::type might_call;
    might_call(this,id,rp,c, mcc);
    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(RunPrincipal& rp, EventSetup const& c,
                        ModuleCallingContext const* mcc) {
    module_->doEndRun(rp, c, mcc);
    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoBegin(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                          ModuleCallingContext const* mcc) {
    module_->doBeginLuminosityBlock(lbp, c, mcc);
    return true;
  }
  
  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamBegin(D, StreamID id, LuminosityBlockPrincipal& rp,
                                    EventSetup const& c,
                                    ModuleCallingContext const* mcc) {
    module_->doStreamBeginLuminosityBlock(id, rp, c, mcc);
  }
  
  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamEnd(D, StreamID id, LuminosityBlockPrincipal& rp,
                                  EventSetup const& c,
                                  ModuleCallingContext const* mcc) {
    module_->doStreamEndLuminosityBlock(id, rp, c, mcc);
  }

  
  template<typename T>
  inline
  bool
    WorkerT<T>::implDoStreamBegin(StreamID id, LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                  ModuleCallingContext const* mcc) {
    typename boost::mpl::if_c<workerimpl::has_stream_functions<T>::value,
    workerimpl::DoStreamBeginTrans<T,LuminosityBlockPrincipal>,
    workerimpl::DoNothing>::type might_call;
    might_call(this,id,lbp,c, mcc);
    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoStreamEnd(StreamID id, LuminosityBlockPrincipal& lbp, EventSetup const& c,
                              ModuleCallingContext const* mcc) {
    typename boost::mpl::if_c<workerimpl::has_stream_functions<T>::value,
    workerimpl::DoStreamEndTrans<T,LuminosityBlockPrincipal>,
    workerimpl::DoNothing>::type might_call;
    might_call(this,id,lbp,c,mcc);

    return true;
  }
  
  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                        ModuleCallingContext const* mcc) {
    module_->doEndLuminosityBlock(lbp, c, mcc);
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
  template<typename D>
  void WorkerT<T>::callWorkerBeginStream(D, StreamID id) {
    module_->doBeginStream(id);
  }

  template<typename T>
  inline
  void
  WorkerT<T>::implBeginStream(StreamID id) {
    typename boost::mpl::if_c<workerimpl::has_stream_functions<T>::value,
    workerimpl::DoBeginStream<T>,
    workerimpl::DoNothing>::type might_call;
    might_call(this,id);
  }
  
  template<typename T>
  template<typename D>
  void WorkerT<T>::callWorkerEndStream(D, StreamID id) {
    module_->doEndStream(id);
  }
  
  template<typename T>
  inline
  void
  WorkerT<T>::implEndStream(StreamID id) {
    typename boost::mpl::if_c<workerimpl::has_stream_functions<T>::value,
    workerimpl::DoEndStream<T>,
    workerimpl::DoNothing>::type might_call;
    might_call(this,id);
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
  void WorkerT<T>::updateLookup(BranchType iBranchType,
                                ProductHolderIndexHelper const& iHelper) {
    module_->updateLookup(iBranchType,iHelper);
  }

  template<>
  Worker::Types WorkerT<EDAnalyzer>::moduleType() const { return Worker::kAnalyzer;}
  template<>
  Worker::Types WorkerT<EDProducer>::moduleType() const { return Worker::kProducer;}
  template<>
  Worker::Types WorkerT<EDFilter>::moduleType() const { return Worker::kFilter;}
  template<>
  Worker::Types WorkerT<OutputModule>::moduleType() const { return Worker::kOutputModule;}
  template<>
  Worker::Types WorkerT<edm::one::EDProducerBase>::moduleType() const { return Worker::kProducer;}
  template<>
  Worker::Types WorkerT<edm::one::EDFilterBase>::moduleType() const { return Worker::kFilter;}
  template<>
  Worker::Types WorkerT<edm::one::EDAnalyzerBase>::moduleType() const { return Worker::kAnalyzer;}
  template<>
  Worker::Types WorkerT<edm::one::OutputModuleBase>::moduleType() const { return Worker::kOutputModule;}

  template<>
  Worker::Types WorkerT<edm::global::EDProducerBase>::moduleType() const { return Worker::kProducer;}
  template<>
  Worker::Types WorkerT<edm::global::EDFilterBase>::moduleType() const { return Worker::kFilter;}
  template<>
  Worker::Types WorkerT<edm::global::EDAnalyzerBase>::moduleType() const { return Worker::kAnalyzer;}


  template<>
  Worker::Types WorkerT<edm::stream::EDProducerAdaptorBase>::moduleType() const { return Worker::kProducer;}
  template<>
  Worker::Types WorkerT<edm::stream::EDFilterAdaptorBase>::moduleType() const { return Worker::kFilter;}
  template<>
  Worker::Types WorkerT<edm::stream::EDAnalyzerAdaptorBase>::moduleType() const { return Worker::kAnalyzer;}

  //Explicitly instantiate our needed templates to avoid having the compiler
  // instantiate them in all of our libraries
  template class WorkerT<EDProducer>;
  template class WorkerT<EDFilter>;
  template class WorkerT<EDAnalyzer>;
  template class WorkerT<OutputModule>;
  template class WorkerT<one::EDProducerBase>;
  template class WorkerT<one::EDFilterBase>;
  template class WorkerT<one::EDAnalyzerBase>;
  template class WorkerT<one::OutputModuleBase>;
  template class WorkerT<global::EDProducerBase>;
  template class WorkerT<global::EDFilterBase>;
  template class WorkerT<global::EDAnalyzerBase>;
  template class WorkerT<stream::EDProducerAdaptorBase>;
  template class WorkerT<stream::EDFilterAdaptorBase>;
  template class WorkerT<stream::EDAnalyzerAdaptorBase>;
}
