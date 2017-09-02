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
#include "FWCore/Framework/interface/global/OutputModuleBase.h"

#include "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDFilterAdaptorBase.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerAdaptorBase.h"

#include "FWCore/Framework/interface/limited/EDProducerBase.h"
#include "FWCore/Framework/interface/limited/EDFilterBase.h"
#include "FWCore/Framework/interface/limited/EDAnalyzerBase.h"
#include "FWCore/Framework/interface/limited/OutputModuleBase.h"

#include <type_traits>

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
    struct has_stream_functions<edm::limited::EDProducerBase> {
      static bool constexpr value = true;
    };
    
    template<>
    struct has_stream_functions<edm::limited::EDFilterBase> {
      static bool constexpr value = true;
    };
    
    template<>
    struct has_stream_functions<edm::limited::EDAnalyzerBase> {
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
      inline void operator() (WorkerT<T>* iWorker, StreamID id, P const& rp,
                              EventSetup const& c,
                              ModuleCallingContext const* mcc) {
        iWorker->callWorkerStreamBegin(0,id,rp,c, mcc);
      }
    };

    template<typename T, typename P>
    struct DoStreamEndTrans {
      inline void operator() (WorkerT<T>* iWorker, StreamID id, P const& rp,
                              EventSetup const& c,
                              ModuleCallingContext const* mcc) {
        iWorker->callWorkerStreamEnd(0,id,rp,c, mcc);
      }
    };
  }

  template<typename T>
  inline
  WorkerT<T>::WorkerT(std::shared_ptr<T> ed, ModuleDescription const& md, ExceptionToActionTable const* actions) :
    Worker(md, actions),
    module_(ed) {
    assert(module_ != nullptr);
  }

  template<typename T>
  WorkerT<T>::~WorkerT() {
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDo(EventPrincipal const& ep, EventSetup const& c, ModuleCallingContext const* mcc) {
    std::shared_ptr<Worker> sentry(this,[&ep](Worker* obj) {obj->postDoEvent(ep);});
    return module_->doEvent(ep, c, activityRegistry(), mcc);
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoPrePrefetchSelection(StreamID id,
                                         EventPrincipal const& ep,
                                         ModuleCallingContext const* mcc) {
    return true;
  }

  template<>
  inline
  bool
  WorkerT<OutputModule>::implDoPrePrefetchSelection(StreamID id,
                                                    EventPrincipal const& ep,
                                                    ModuleCallingContext const* mcc) {
    return module_->prePrefetchSelection(id,ep,mcc);
  }

  template<>
  inline
  bool
  WorkerT<edm::one::OutputModuleBase>::implDoPrePrefetchSelection(StreamID id,
                                                                  EventPrincipal const& ep,
                                                                  ModuleCallingContext const* mcc) {
    return module_->prePrefetchSelection(id,ep,mcc);
  }

  template<>
  inline
  bool
  WorkerT<edm::global::OutputModuleBase>::implDoPrePrefetchSelection(StreamID id,
                                                                     EventPrincipal const& ep,
                                                                     ModuleCallingContext const* mcc) {
    return module_->prePrefetchSelection(id,ep,mcc);
  }

  template<>
  inline
  bool
  WorkerT<edm::limited::OutputModuleBase>::implDoPrePrefetchSelection(StreamID id,
                                                                     EventPrincipal const& ep,
                                                                     ModuleCallingContext const* mcc) {
    return module_->prePrefetchSelection(id,ep,mcc);
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoBegin(RunPrincipal const& rp, EventSetup const& c, ModuleCallingContext const* mcc) {
    module_->doBeginRun(rp, c, mcc);
    return true;
  }

  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamBegin(D, StreamID id, RunPrincipal const& rp,
                                    EventSetup const& c,
                                    ModuleCallingContext const* mcc) {
    module_->doStreamBeginRun(id, rp, c, mcc);
  }

  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamEnd(D, StreamID id, RunPrincipal const& rp,
                                  EventSetup const& c,
                                  ModuleCallingContext const* mcc) {
    module_->doStreamEndRun(id, rp, c, mcc);
  }


  template<typename T>
  inline
  bool
  WorkerT<T>::implDoStreamBegin(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                                ModuleCallingContext const* mcc) {
    std::conditional_t<workerimpl::has_stream_functions<T>::value,
                       workerimpl::DoStreamBeginTrans<T,RunPrincipal const>,
                       workerimpl::DoNothing> might_call;
    might_call(this,id,rp,c, mcc);
    return true;
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoStreamEnd(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                              ModuleCallingContext const* mcc) {
    std::conditional_t<workerimpl::has_stream_functions<T>::value,
                       workerimpl::DoStreamEndTrans<T,RunPrincipal const>,
                       workerimpl::DoNothing> might_call;
    might_call(this,id,rp,c, mcc);
    return true;
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(RunPrincipal const& rp, EventSetup const& c,
                        ModuleCallingContext const* mcc) {
    module_->doEndRun(rp, c, mcc);
    return true;
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoBegin(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                          ModuleCallingContext const* mcc) {
    module_->doBeginLuminosityBlock(lbp, c, mcc);
    return true;
  }

  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamBegin(D, StreamID id, LuminosityBlockPrincipal const& rp,
                                    EventSetup const& c,
                                    ModuleCallingContext const* mcc) {
    module_->doStreamBeginLuminosityBlock(id, rp, c, mcc);
  }

  template<typename T>
  template<typename D>
  void
  WorkerT<T>::callWorkerStreamEnd(D, StreamID id, LuminosityBlockPrincipal const& rp,
                                  EventSetup const& c,
                                  ModuleCallingContext const* mcc) {
    module_->doStreamEndLuminosityBlock(id, rp, c, mcc);
  }


  template<typename T>
  inline
  bool
  WorkerT<T>::implDoStreamBegin(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                ModuleCallingContext const* mcc) {
    std::conditional_t<workerimpl::has_stream_functions<T>::value,
                       workerimpl::DoStreamBeginTrans<T,LuminosityBlockPrincipal>,
                       workerimpl::DoNothing> might_call;
    might_call(this,id,lbp,c, mcc);
    return true;
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoStreamEnd(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                              ModuleCallingContext const* mcc) {
    std::conditional_t<workerimpl::has_stream_functions<T>::value,
                       workerimpl::DoStreamEndTrans<T,LuminosityBlockPrincipal>,
                       workerimpl::DoNothing> might_call;
    might_call(this,id,lbp,c,mcc);

    return true;
  }

  template<typename T>
  inline
  bool
  WorkerT<T>::implDoEnd(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
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
    std::conditional_t<workerimpl::has_stream_functions<T>::value,
                       workerimpl::DoBeginStream<T>,
                       workerimpl::DoNothing> might_call;
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
    std::conditional_t<workerimpl::has_stream_functions<T>::value,
                       workerimpl::DoEndStream<T>,
                       workerimpl::DoNothing> might_call;
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
  WorkerT<T>::implRegisterThinnedAssociations(ProductRegistry const& registry,
                                              ThinnedAssociationsHelper& helper) {
    module_->doRegisterThinnedAssociations(registry, helper);
  }

  template<typename T>
  inline
  Worker::TaskQueueAdaptor WorkerT<T>::serializeRunModule() {
    return Worker::TaskQueueAdaptor{};
  }
  template<> Worker::TaskQueueAdaptor WorkerT<EDAnalyzer>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<EDFilter>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<EDProducer>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<OutputModule>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<one::EDAnalyzerBase>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<one::EDFilterBase>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<one::EDProducerBase>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<one::OutputModuleBase>::serializeRunModule() {
    return &(module_->sharedResourcesAcquirer().serialQueueChain());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<limited::EDAnalyzerBase>::serializeRunModule() {
    return &(module_->queue());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<limited::EDFilterBase>::serializeRunModule() {
    return &(module_->queue());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<limited::EDProducerBase>::serializeRunModule() {
    return &(module_->queue());
  }
  template<> Worker::TaskQueueAdaptor WorkerT<limited::OutputModuleBase>::serializeRunModule() {
    return &(module_->queue());
  }


  namespace {
    template <typename T> bool mustPrefetchMayGet();

    template<> bool mustPrefetchMayGet<EDAnalyzer>() { return true;}
    template<> bool mustPrefetchMayGet<EDProducer>() { return true;}
    template<> bool mustPrefetchMayGet<EDFilter>() { return true;}
    template<> bool mustPrefetchMayGet<OutputModule>() { return true;}

    template<> bool mustPrefetchMayGet<edm::one::EDProducerBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::one::EDFilterBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::one::EDAnalyzerBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::one::OutputModuleBase>() { return true;}

    template<> bool mustPrefetchMayGet<edm::global::EDProducerBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::global::EDFilterBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::global::EDAnalyzerBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::global::OutputModuleBase>() { return true;}

    template<> bool mustPrefetchMayGet<edm::limited::EDProducerBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::limited::EDFilterBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::limited::EDAnalyzerBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::limited::OutputModuleBase>() { return true;}

    template<> bool mustPrefetchMayGet<edm::stream::EDProducerAdaptorBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::stream::EDFilterAdaptorBase>() { return true;}
    template<> bool mustPrefetchMayGet<edm::stream::EDAnalyzerAdaptorBase>() { return true;}

  }


  template<typename T>
  void WorkerT<T>::updateLookup(BranchType iBranchType,
                                ProductResolverIndexHelper const& iHelper) {
    module_->updateLookup(iBranchType,iHelper,mustPrefetchMayGet<T>());
  }

  namespace {
    void resolvePutIndiciesImpl(void*,
                                BranchType iBranchType,
                                std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies,
                                std::string const& iModuleLabel) {
      //Do nothing
    }

    void resolvePutIndiciesImpl(ProducerBase* iProd,
                                BranchType iBranchType,
                                std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies,
                                std::string const& iModuleLabel) {
      iProd->resolvePutIndicies(iBranchType, iIndicies, iModuleLabel);
    }

    void resolvePutIndiciesImpl(edm::stream::EDProducerAdaptorBase* iProd,
                                BranchType iBranchType,
                                std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies,
                                std::string const& iModuleLabel) {
      iProd->resolvePutIndicies(iBranchType, iIndicies, iModuleLabel);
    }
    void resolvePutIndiciesImpl(edm::stream::EDFilterAdaptorBase* iProd,
                                BranchType iBranchType,
                                std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies,
                                std::string const& iModuleLabel) {
      iProd->resolvePutIndicies(iBranchType, iIndicies, iModuleLabel);
    }

    std::vector<ProductResolverIndex> s_emptyIndexList;

    std::vector<ProductResolverIndex> const& itemsShouldPutInEventImpl(void const*) {
      return s_emptyIndexList;
    }

    std::vector<ProductResolverIndex> const& itemsShouldPutInEventImpl(ProducerBase const* iProd) {
      return iProd->indiciesForPutProducts(edm::InEvent);
    }

    std::vector<ProductResolverIndex> const& itemsShouldPutInEventImpl(edm::stream::EDProducerAdaptorBase const* iProd) {
      return iProd->indiciesForPutProducts(edm::InEvent);
    }

    std::vector<ProductResolverIndex> const& itemsShouldPutInEventImpl(edm::stream::EDFilterAdaptorBase const* iProd) {
      return iProd->indiciesForPutProducts(edm::InEvent);
    }

  }

  template<typename T>
  void WorkerT<T>::resolvePutIndicies(BranchType iBranchType,
                                      std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies) {
    resolvePutIndiciesImpl(&module(), iBranchType,iIndicies, description().moduleLabel());
  }


  template<typename T>
  std::vector<ProductResolverIndex> const&
  WorkerT<T>::itemsShouldPutInEvent() const {
    return itemsShouldPutInEventImpl(&module());
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
  Worker::Types WorkerT<edm::global::OutputModuleBase>::moduleType() const { return Worker::kOutputModule;}

  template<>
  Worker::Types WorkerT<edm::limited::EDProducerBase>::moduleType() const { return Worker::kProducer;}
  template<>
  Worker::Types WorkerT<edm::limited::EDFilterBase>::moduleType() const { return Worker::kFilter;}
  template<>
  Worker::Types WorkerT<edm::limited::EDAnalyzerBase>::moduleType() const { return Worker::kAnalyzer;}
  template<>
  Worker::Types WorkerT<edm::limited::OutputModuleBase>::moduleType() const { return Worker::kOutputModule;}


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
  template class WorkerT<global::OutputModuleBase>;
  template class WorkerT<stream::EDProducerAdaptorBase>;
  template class WorkerT<stream::EDFilterAdaptorBase>;
  template class WorkerT<stream::EDAnalyzerAdaptorBase>;
  template class WorkerT<limited::EDProducerBase>;
  template class WorkerT<limited::EDFilterBase>;
  template class WorkerT<limited::EDAnalyzerBase>;
  template class WorkerT<limited::OutputModuleBase>;
}
