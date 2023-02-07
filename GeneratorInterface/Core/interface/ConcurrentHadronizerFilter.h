// -*- C++ -*-
//
//

// class template ConcurrentHadronizerFilter<HAD> provides an EDFilter which uses
// the hadronizer type HAD to read in external partons and hadronize them,
// and decay the resulting particles, in the CMS framework.

// Some additional notes related to concurrency:
//
//     This is an unusual module in CMSSW because its hadronizers are in stream caches
//     (one hadronizer per stream cache). The Framework expects objects in a stream
//     cache to only be used in stream transitions associated with that stream. That
//     is how the Framework provides thread safety and avoids data races. In this module
//     a global transition needs to use one of the hadronizers. The
//     globalBeginLuminosityBlockProduce method uses one hadronizer to create the
//     GenLumiInfoHeader which is put in the LuminosityBlock. This hadronizer must be
//     initialized for the lumi before creating the product. This creates a problem because
//     the global method might run concurrently with the stream methods. There is extra
//     complexity in this module to deal with that unusual usage of an object in a stream cache.
//
//     The solution of this issue is conceptually simple. The module explicitly makes
//     globalBeginLuminosityBlock wait until the previous lumi is finished on one stream
//     and also until streamEndRun is finished on that stream if there was a new run. It
//     avoids doing work in streamBeginRun. There is extra complexity in this module to
//     ensure thread safety that normally does not appear in modules (usually this kind of
//     thing is handled in the Framework).
//
//     Two alternative solutions were considered when designing this implementation and
//     possibly someday we might reimplement this using one of them if we find this
//     complexity hard to maintain.
//
//     1. We could make an extra hadronizer only for the global transition. We rejected
//     that idea because that would require extra memory and CPU resources.
//
//     2. We could put the GenLumiInfoHeader product into the LuminosityBlock at the end
//     global transition. We didn't know whether anything depended on the product being
//     present in the begin transition or how difficult it would be to remove such a dependence
//     so we also rejected that alternative.
//
//     There might be other ways to deal with this concurrency issue. This issue became
//     important when run concurrency support was implemented in the Framework. That support
//     allowed the streamBeginRun and streamEndRun transitions to run concurrently with other
//     transitions even in the case where the number of concurrent runs was limited to 1.

#ifndef GeneratorInterface_Core_ConcurrentHadronizerFilter_h
#define GeneratorInterface_Core_ConcurrentHadronizerFilter_h

#include <memory>
#include <string>
#include <vector>
#include <atomic>

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "CLHEP/Random/RandomEngine.h"

#include "GeneratorInterface/Core/interface/HepMCFilterDriver.h"

// LHE Run
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

namespace edm {
  namespace gen {
    struct RunCache {
      mutable std::atomic<GenRunInfoProduct*> product_{nullptr};
      ~RunCache() { delete product_.load(); }

      //This is called from globalEndRunProduce which is known to
      // be safe as the framework would not be calling any other
      // methods of this module using this run at that time
      std::unique_ptr<GenRunInfoProduct> release() const noexcept {
        auto retValue = product_.load();
        product_.store(nullptr);
        return std::unique_ptr<GenRunInfoProduct>(retValue);
      }
    };
    struct LumiSummary {
      mutable std::unique_ptr<GenLumiInfoProduct> lumiInfo_;
      mutable std::unique_ptr<GenFilterInfo> filterInfo_;
    };
    template <typename HAD, typename DEC>
    struct StreamCache {
      StreamCache(ParameterSet const& iPSet) : hadronizer_{iPSet} {}
      HAD hadronizer_;
      std::unique_ptr<DEC> decayer_;
      std::unique_ptr<HepMCFilterDriver> filter_;
      unsigned long long nInitializedWithLHERunInfo_{0};
      unsigned long long nStreamEndLumis_{0};
      bool initialized_ = false;
    };
    template <typename HAD, typename DEC>
    struct LumiCache {
      gen::StreamCache<HAD, DEC>* useInLumi_{nullptr};
      unsigned long long nGlobalBeginRuns_{0};
    };
  }  // namespace gen

  template <class HAD, class DEC>
  class ConcurrentHadronizerFilter : public global::EDFilter<EndRunProducer,
                                                             BeginLuminosityBlockProducer,
                                                             EndLuminosityBlockProducer,
                                                             RunCache<gen::RunCache>,
                                                             LuminosityBlockCache<gen::LumiCache<HAD, DEC>>,
                                                             LuminosityBlockSummaryCache<gen::LumiSummary>,
                                                             StreamCache<gen::StreamCache<HAD, DEC>>> {
  public:
    typedef HAD Hadronizer;
    typedef DEC Decayer;

    // The given ParameterSet will be passed to the contained
    // Hadronizer object.
    explicit ConcurrentHadronizerFilter(ParameterSet const& ps);

    std::unique_ptr<gen::StreamCache<HAD, DEC>> beginStream(edm::StreamID) const override;
    bool filter(StreamID id, Event& e, EventSetup const& es) const override;
    void streamEndRun(StreamID, Run const&, EventSetup const&) const override;
    std::shared_ptr<gen::RunCache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override;
    void globalEndRunProduce(Run&, EventSetup const&) const override;
    void streamBeginLuminosityBlock(StreamID, LuminosityBlock const&, EventSetup const&) const override;
    std::shared_ptr<gen::LumiCache<HAD, DEC>> globalBeginLuminosityBlock(LuminosityBlock const&,
                                                                         EventSetup const&) const override;
    void globalBeginLuminosityBlockProduce(LuminosityBlock&, EventSetup const&) const override;
    void streamEndLuminosityBlockSummary(StreamID,
                                         LuminosityBlock const&,
                                         EventSetup const&,
                                         gen::LumiSummary*) const override;
    std::shared_ptr<gen::LumiSummary> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                        edm::EventSetup const&) const override;
    void globalEndLuminosityBlock(LuminosityBlock const&, EventSetup const&) const override;
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         gen::LumiSummary*) const override;
    void globalEndLuminosityBlockProduce(LuminosityBlock&, EventSetup const&, gen::LumiSummary const*) const override;

  private:
    void initializeWithLHERunInfo(gen::StreamCache<HAD, DEC>*, LuminosityBlock const&) const;
    void initLumi(gen::StreamCache<HAD, DEC>* cache, LuminosityBlock const& index, EventSetup const& es) const;
    ParameterSet config_;
    InputTag runInfoProductTag_;
    EDGetTokenT<LHERunInfoProduct> runInfoProductToken_;
    EDGetTokenT<LHEEventProduct> eventProductToken_;
    unsigned int counterRunInfoProducts_;
    unsigned int nAttempts_;
    mutable std::atomic<gen::StreamCache<HAD, DEC>*> useInLumi_{nullptr};
    mutable std::atomic<unsigned long long> greatestNStreamEndLumis_{0};
    mutable std::atomic<bool> streamEndRunComplete_{true};
    // The next two data members are thread safe and can be safely mutable because
    // they are only modified/read in globalBeginRun and globalBeginLuminosityBlock.
    mutable unsigned long long nGlobalBeginRuns_{0};
    mutable unsigned long long nInitializedWithLHERunInfo_{0};
    bool const hasFilter_;
  };

  //------------------------------------------------------------------------
  //
  // Implementation

  template <class HAD, class DEC>
  ConcurrentHadronizerFilter<HAD, DEC>::ConcurrentHadronizerFilter(ParameterSet const& ps)
      : config_(ps),
        runInfoProductTag_(),
        runInfoProductToken_(),
        eventProductToken_(),
        counterRunInfoProducts_(0),
        nAttempts_(1),
        hasFilter_(ps.exists("HepMCFilter")) {
    auto ptrThis = this;
    this->callWhenNewProductsRegistered([ptrThis](BranchDescription const& iBD) {
      //this is called each time a module registers that it will produce a LHERunInfoProduct
      if (iBD.unwrappedTypeID() == edm::TypeID(typeid(LHERunInfoProduct)) && iBD.branchType() == InRun) {
        ++(ptrThis->counterRunInfoProducts_);
        ptrThis->eventProductToken_ = ptrThis->template consumes<LHEEventProduct>(
            InputTag((iBD.moduleLabel() == "externalLHEProducer") ? "externalLHEProducer" : "source"));
        ptrThis->runInfoProductTag_ = InputTag(iBD.moduleLabel(), iBD.productInstanceName(), iBD.processName());
        ptrThis->runInfoProductToken_ = ptrThis->template consumes<LHERunInfoProduct, InRun>(
            InputTag(iBD.moduleLabel(), iBD.productInstanceName(), iBD.processName()));
      }
    });

    // TODO:
    // Put the list of types produced by the filters here.
    // The current design calls for:
    //   * LHEGeneratorInfo
    //   * LHEEvent
    //   * HepMCProduct
    // But I can not find the LHEGeneratorInfo class; it might need to
    // be invented.

    //initialize setting for multiple hadronization attempts
    if (ps.exists("nAttempts")) {
      nAttempts_ = ps.getParameter<unsigned int>("nAttempts");
    }

    this->template produces<edm::HepMCProduct>("unsmeared");
    this->template produces<GenEventInfoProduct>();
    this->template produces<GenLumiInfoHeader, edm::Transition::BeginLuminosityBlock>();
    this->template produces<GenLumiInfoProduct, edm::Transition::EndLuminosityBlock>();
    this->template produces<GenRunInfoProduct, edm::Transition::EndRun>();
    if (hasFilter_)
      this->template produces<GenFilterInfo, edm::Transition::EndLuminosityBlock>();
  }

  template <class HAD, class DEC>
  std::unique_ptr<gen::StreamCache<HAD, DEC>> ConcurrentHadronizerFilter<HAD, DEC>::beginStream(edm::StreamID) const {
    auto cache = std::make_unique<gen::StreamCache<HAD, DEC>>(config_);

    if (config_.exists("ExternalDecays")) {
      ParameterSet ps1 = config_.getParameter<ParameterSet>("ExternalDecays");
      cache->decayer_.reset(new Decayer(ps1));
    }

    if (config_.exists("HepMCFilter")) {
      ParameterSet psfilter = config_.getParameter<ParameterSet>("HepMCFilter");
      cache->filter_.reset(new HepMCFilterDriver(psfilter));
    }

    //We need a hadronizer during globalBeginLumiProduce, doesn't matter which one
    gen::StreamCache<HAD, DEC>* expected = nullptr;
    useInLumi_.compare_exchange_strong(expected, cache.get());

    return cache;
  }

  template <class HAD, class DEC>
  bool ConcurrentHadronizerFilter<HAD, DEC>::filter(StreamID id, Event& ev, EventSetup const& /* es */) const {
    auto cache = this->streamCache(id);
    RandomEngineSentry<HAD> randomEngineSentry(&cache->hadronizer_, ev.streamID());
    RandomEngineSentry<DEC> randomEngineSentryDecay(cache->decayer_.get(), ev.streamID());

    cache->hadronizer_.setEDMEvent(ev);

    // get LHE stuff and pass to hadronizer!
    //
    edm::Handle<LHEEventProduct> product;
    ev.getByToken(eventProductToken_, product);

    std::unique_ptr<HepMC::GenEvent> finalEvent;
    std::unique_ptr<GenEventInfoProduct> finalGenEventInfo;

    //number of accepted events
    unsigned int naccept = 0;

    for (unsigned int itry = 0; itry < nAttempts_; ++itry) {
      cache->hadronizer_.setLHEEvent(std::make_unique<lhef::LHEEvent>(cache->hadronizer_.getLHERunInfo(), *product));

      // cache->hadronizer_.generatePartons();
      if (!cache->hadronizer_.hadronize())
        continue;

      //  this is "fake" stuff
      // in principle, decays are done as part of full event generation,
      // except for particles that are marked as to be kept stable
      // but we currently keep in it the design, because we might want
      // to use such feature for other applications
      //
      if (!cache->hadronizer_.decay())
        continue;

      std::unique_ptr<HepMC::GenEvent> event(cache->hadronizer_.getGenEvent());
      if (!event.get())
        continue;

      // The external decay driver is being added to the system,
      // it should be called here
      //
      if (cache->decayer_) {
        auto lheEvent = cache->hadronizer_.getLHEEvent();
        auto t = cache->decayer_->decay(event.get(), lheEvent.get());
        if (t != event.get()) {
          event.reset(t);
        }
        cache->hadronizer_.setLHEEvent(std::move(lheEvent));
      }

      if (!event.get())
        continue;

      // check and perform if there're any unstable particles after
      // running external decay packges
      //
      cache->hadronizer_.resetEvent(std::move(event));
      if (!cache->hadronizer_.residualDecay())
        continue;

      cache->hadronizer_.finalizeEvent();

      event = cache->hadronizer_.getGenEvent();
      if (!event.get())
        continue;

      event->set_event_number(ev.id().event());

      std::unique_ptr<GenEventInfoProduct> genEventInfo(cache->hadronizer_.getGenEventInfo());
      if (!genEventInfo.get()) {
        // create GenEventInfoProduct from HepMC event in case hadronizer didn't provide one
        genEventInfo = std::make_unique<GenEventInfoProduct>(event.get());
      }

      //if HepMCFilter was specified, test event
      if (cache->filter_ && !cache->filter_->filter(event.get(), genEventInfo->weight()))
        continue;

      ++naccept;

      //keep the LAST accepted event (which is equivalent to choosing randomly from the accepted events)
      finalEvent = std::move(event);
      finalGenEventInfo = std::move(genEventInfo);
    }

    if (!naccept)
      return false;

    //adjust event weights if necessary (in case input event was attempted multiple times)
    if (nAttempts_ > 1) {
      double multihadweight = double(naccept) / double(nAttempts_);

      //adjust weight for GenEventInfoProduct
      finalGenEventInfo->weights()[0] *= multihadweight;

      //adjust weight for HepMC GenEvent (used e.g for RIVET)
      finalEvent->weights()[0] *= multihadweight;
    }

    ev.put(std::move(finalGenEventInfo));

    std::unique_ptr<HepMCProduct> bare_product(new HepMCProduct());
    bare_product->addHepMCData(finalEvent.release());
    ev.put(std::move(bare_product), "unsmeared");

    return true;
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::streamEndRun(StreamID id, Run const& r, EventSetup const&) const {
    auto rCache = this->runCache(r.index());
    auto cache = this->streamCache(id);

    // Retrieve the LHE run info summary and transfer determined
    // cross-section into the generator run info

    const lhef::LHERunInfo* lheRunInfo = cache->hadronizer_.getLHERunInfo().get();
    lhef::LHERunInfo::XSec xsec = lheRunInfo->xsec();

    GenRunInfoProduct& genRunInfo = cache->hadronizer_.getGenRunInfo();
    genRunInfo.setInternalXSec(GenRunInfoProduct::XSec(xsec.value(), xsec.error()));

    // If relevant, record the integrated luminosity for this run
    // here.  To do so, we would need a standard function to invoke on
    // the contained hadronizer that would report the integrated
    // luminosity.

    if (cache->initialized_) {
      cache->hadronizer_.statistics();
      if (cache->decayer_)
        cache->decayer_->statistics();
      if (cache->filter_)
        cache->filter_->statistics();
      lheRunInfo->statistics();
    }
    GenRunInfoProduct* expect = nullptr;

    std::unique_ptr<GenRunInfoProduct> griproduct(new GenRunInfoProduct(genRunInfo));
    //All the GenRunInfoProducts for all streams shoule be identical, therefore we only
    // need one
    if (rCache->product_.compare_exchange_strong(expect, griproduct.get())) {
      griproduct.release();
    }
    if (cache == useInLumi_.load()) {
      streamEndRunComplete_ = true;
    }
  }

  template <class HAD, class DEC>
  std::shared_ptr<gen::RunCache> ConcurrentHadronizerFilter<HAD, DEC>::globalBeginRun(edm::Run const&,
                                                                                      edm::EventSetup const&) const {
    ++nGlobalBeginRuns_;

    if (counterRunInfoProducts_ > 1)
      throw edm::Exception(errors::EventCorruption) << "More than one LHERunInfoProduct present";

    if (counterRunInfoProducts_ == 0)
      throw edm::Exception(errors::EventCorruption) << "No LHERunInfoProduct present";

    return std::make_shared<gen::RunCache>();
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::globalEndRun(edm::Run const&, edm::EventSetup const&) const {}

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::globalEndRunProduce(Run& r, EventSetup const&) const {
    r.put(this->runCache(r.index())->release());
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::streamBeginLuminosityBlock(StreamID id,
                                                                        LuminosityBlock const& lumi,
                                                                        EventSetup const& es) const {
    gen::StreamCache<HAD, DEC>* streamCachePtr = this->streamCache(id);
    bool newRun =
        streamCachePtr->nInitializedWithLHERunInfo_ < this->luminosityBlockCache(lumi.index())->nGlobalBeginRuns_;
    if (newRun) {
      streamCachePtr->nInitializedWithLHERunInfo_ = this->luminosityBlockCache(lumi.index())->nGlobalBeginRuns_;
    }
    if (this->luminosityBlockCache(lumi.index())->useInLumi_ != streamCachePtr) {
      if (newRun) {
        initializeWithLHERunInfo(streamCachePtr, lumi);
      }
      initLumi(streamCachePtr, lumi, es);
    }
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::initializeWithLHERunInfo(gen::StreamCache<HAD, DEC>* streamCachePtr,
                                                                      edm::LuminosityBlock const& lumi) const {
    edm::Handle<LHERunInfoProduct> lheRunInfoProduct;
    lumi.getRun().getByLabel(runInfoProductTag_, lheRunInfoProduct);
    //TODO: fix so that this actually works with getByToken commented below...
    //run.getByToken(runInfoProductToken_, lheRunInfoProduct);
    auto& hadronizer = streamCachePtr->hadronizer_;

    hadronizer.setLHERunInfo(std::make_unique<lhef::LHERunInfo>(*lheRunInfoProduct));
    lhef::LHERunInfo* lheRunInfo = hadronizer.getLHERunInfo().get();
    lheRunInfo->initLumi();
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::initLumi(gen::StreamCache<HAD, DEC>* cache,
                                                      LuminosityBlock const& lumi,
                                                      EventSetup const& es) const {
    lhef::LHERunInfo* lheRunInfo = cache->hadronizer_.getLHERunInfo().get();
    lheRunInfo->initLumi();

    //We need all copies to see same random # for begin lumi
    Service<RandomNumberGenerator> rng;
    auto enginePtr = rng->cloneEngine(lumi.index());
    cache->hadronizer_.setRandomEngine(enginePtr.get());
    if (cache->decayer_) {
      cache->decayer_->setRandomEngine(enginePtr.get());
    }

    auto unsetH = [](HAD* h) { h->setRandomEngine(nullptr); };
    auto unsetD = [](DEC* d) {
      if (d) {
        d->setRandomEngine(nullptr);
      }
    };

    std::unique_ptr<HAD, decltype(unsetH)> randomEngineSentry(&cache->hadronizer_, unsetH);
    std::unique_ptr<DEC, decltype(unsetD)> randomEngineSentryDecay(cache->decayer_.get(), unsetD);

    cache->hadronizer_.randomizeIndex(lumi, enginePtr.get());

    if (!cache->hadronizer_.readSettings(1))
      throw edm::Exception(errors::Configuration)
          << "Failed to read settings for the hadronizer " << cache->hadronizer_.classname() << " \n";

    if (cache->decayer_) {
      cache->decayer_->init(es);
      if (!cache->hadronizer_.declareStableParticles(cache->decayer_->operatesOnParticles()))
        throw edm::Exception(errors::Configuration)
            << "Failed to declare stable particles in hadronizer " << cache->hadronizer_.classname()
            << " for internal parton generation\n";
      if (!cache->hadronizer_.declareSpecialSettings(cache->decayer_->specialSettings()))
        throw edm::Exception(errors::Configuration)
            << "Failed to declare special settings in hadronizer " << cache->hadronizer_.classname() << "\n";
    }

    if (cache->filter_) {
      cache->filter_->resetStatistics();
    }

    if (!cache->hadronizer_.initializeForExternalPartons())
      throw edm::Exception(errors::Configuration)
          << "Failed to initialize hadronizer " << cache->hadronizer_.classname()
          << " for external parton generation\n";

    cache->initialized_ = true;
  }

  template <class HAD, class DEC>
  std::shared_ptr<gen::LumiCache<HAD, DEC>> ConcurrentHadronizerFilter<HAD, DEC>::globalBeginLuminosityBlock(
      edm::LuminosityBlock const& lumi, edm::EventSetup const&) const {
    //need one of the streams to finish
    while (useInLumi_.load() == nullptr) {
    }

    // streamEndRun also uses the hadronizer in the stream cache
    // so we also need to wait for it to finish if there is a new run
    if (nInitializedWithLHERunInfo_ < nGlobalBeginRuns_) {
      while (!streamEndRunComplete_.load()) {
      }
      nInitializedWithLHERunInfo_ = nGlobalBeginRuns_;
      initializeWithLHERunInfo(useInLumi_.load(), lumi);
    }

    auto lumiCache = std::make_shared<gen::LumiCache<HAD, DEC>>();
    lumiCache->useInLumi_ = useInLumi_.load();
    lumiCache->nGlobalBeginRuns_ = nGlobalBeginRuns_;
    return lumiCache;
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::globalBeginLuminosityBlockProduce(LuminosityBlock& lumi,
                                                                               EventSetup const& es) const {
    initLumi(useInLumi_, lumi, es);
    std::unique_ptr<GenLumiInfoHeader> genLumiInfoHeader(useInLumi_.load()->hadronizer_.getGenLumiInfoHeader());
    lumi.put(std::move(genLumiInfoHeader));
    useInLumi_.store(nullptr);
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::streamEndLuminosityBlockSummary(StreamID id,
                                                                             LuminosityBlock const&,
                                                                             EventSetup const&,
                                                                             gen::LumiSummary* iSummary) const {
    const lhef::LHERunInfo* lheRunInfo = this->streamCache(id)->hadronizer_.getLHERunInfo().get();

    std::vector<lhef::LHERunInfo::Process> LHELumiProcess = lheRunInfo->getLumiProcesses();
    std::vector<GenLumiInfoProduct::ProcessInfo> GenLumiProcess;
    for (unsigned int i = 0; i < LHELumiProcess.size(); i++) {
      lhef::LHERunInfo::Process thisProcess = LHELumiProcess[i];

      GenLumiInfoProduct::ProcessInfo temp;
      temp.setProcess(thisProcess.process());
      temp.setLheXSec(thisProcess.getLHEXSec().value(), thisProcess.getLHEXSec().error());
      temp.setNPassPos(thisProcess.nPassPos());
      temp.setNPassNeg(thisProcess.nPassNeg());
      temp.setNTotalPos(thisProcess.nTotalPos());
      temp.setNTotalNeg(thisProcess.nTotalNeg());
      temp.setTried(thisProcess.tried().n(), thisProcess.tried().sum(), thisProcess.tried().sum2());
      temp.setSelected(thisProcess.selected().n(), thisProcess.selected().sum(), thisProcess.selected().sum2());
      temp.setKilled(thisProcess.killed().n(), thisProcess.killed().sum(), thisProcess.killed().sum2());
      temp.setAccepted(thisProcess.accepted().n(), thisProcess.accepted().sum(), thisProcess.accepted().sum2());
      temp.setAcceptedBr(thisProcess.acceptedBr().n(), thisProcess.acceptedBr().sum(), thisProcess.acceptedBr().sum2());
      GenLumiProcess.push_back(temp);
    }
    GenLumiInfoProduct genLumiInfo;
    genLumiInfo.setHEPIDWTUP(lheRunInfo->getHEPRUP()->IDWTUP);
    genLumiInfo.setProcessInfo(GenLumiProcess);

    if (iSummary->lumiInfo_) {
      iSummary->lumiInfo_->setHEPIDWTUP(lheRunInfo->getHEPRUP()->IDWTUP);
      iSummary->lumiInfo_->mergeProduct(genLumiInfo);
    } else {
      iSummary->lumiInfo_ = std::make_unique<GenLumiInfoProduct>(std::move(genLumiInfo));
    }

    // produce GenFilterInfo if HepMCFilter is called
    if (hasFilter_) {
      auto filter = this->streamCache(id)->filter_.get();
      GenFilterInfo thisProduct(filter->numEventsPassPos(),
                                filter->numEventsPassNeg(),
                                filter->numEventsTotalPos(),
                                filter->numEventsTotalNeg(),
                                filter->sumpass_w(),
                                filter->sumpass_w2(),
                                filter->sumtotal_w(),
                                filter->sumtotal_w2());
      if (not iSummary->filterInfo_) {
        iSummary->filterInfo_ = std::make_unique<GenFilterInfo>(std::move(thisProduct));
      } else {
        iSummary->filterInfo_->mergeProduct(thisProduct);
      }
    }

    // The next section of code depends on the Framework behavior that the stream
    // lumi transitions are executed for all streams for every lumi even when
    // there are no events for a stream to process.
    gen::StreamCache<HAD, DEC>* streamCachePtr = this->streamCache(id);
    unsigned long long expected = streamCachePtr->nStreamEndLumis_;
    ++streamCachePtr->nStreamEndLumis_;
    if (greatestNStreamEndLumis_.compare_exchange_strong(expected, streamCachePtr->nStreamEndLumis_)) {
      streamEndRunComplete_ = false;
      useInLumi_ = streamCachePtr;
    }
  }

  template <class HAD, class DEC>
  std::shared_ptr<gen::LumiSummary> ConcurrentHadronizerFilter<HAD, DEC>::globalBeginLuminosityBlockSummary(
      edm::LuminosityBlock const&, edm::EventSetup const&) const {
    return std::make_shared<gen::LumiSummary>();
  }

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::globalEndLuminosityBlock(edm::LuminosityBlock const&,
                                                                      edm::EventSetup const&) const {}

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                             edm::EventSetup const&,
                                                                             gen::LumiSummary*) const {}

  template <class HAD, class DEC>
  void ConcurrentHadronizerFilter<HAD, DEC>::globalEndLuminosityBlockProduce(LuminosityBlock& lumi,
                                                                             EventSetup const&,
                                                                             gen::LumiSummary const* iSummary) const {
    //Advance the random number generator so next begin lumi starts with new seed
    Service<RandomNumberGenerator> rng;
    rng->getEngine(lumi.index()).flat();

    lumi.put(std::move(iSummary->lumiInfo_));

    // produce GenFilterInfo if HepMCFilter is called
    if (hasFilter_) {
      lumi.put(std::move(iSummary->filterInfo_));
    }
  }

}  // namespace edm

#endif  // GeneratorInterface_Core_ConcurrentHadronizerFilter_h
