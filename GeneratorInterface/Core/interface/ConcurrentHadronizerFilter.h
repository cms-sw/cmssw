// -*- C++ -*-
//
//

// class template ConcurrentHadronizerFilter<HAD> provides an EDFilter which uses
// the hadronizer type HAD to read in external partons and hadronize them,
// and decay the resulting particles, in the CMS framework.

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
      bool initialized_ = false;
    };
  }  // namespace gen

  template <class HAD, class DEC>
  class ConcurrentHadronizerFilter : public global::EDFilter<EndRunProducer,
                                                             BeginLuminosityBlockProducer,
                                                             EndLuminosityBlockProducer,
                                                             RunCache<gen::RunCache>,
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
    void streamBeginRun(StreamID, Run const&, EventSetup const&) const override;
    void streamEndRun(StreamID, Run const&, EventSetup const&) const override;
    std::shared_ptr<gen::RunCache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override;
    void globalEndRunProduce(Run&, EventSetup const&) const override;
    void streamBeginLuminosityBlock(StreamID, LuminosityBlock const&, EventSetup const&) const override;
    void globalBeginLuminosityBlockProduce(LuminosityBlock&, EventSetup const&) const override;
    void streamEndLuminosityBlockSummary(StreamID,
                                         LuminosityBlock const&,
                                         EventSetup const&,
                                         gen::LumiSummary*) const override;
    std::shared_ptr<gen::LumiSummary> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                        edm::EventSetup const&) const override;
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         gen::LumiSummary*) const override;
    void globalEndLuminosityBlockProduce(LuminosityBlock&, EventSetup const&, gen::LumiSummary const*) const override;

  private:
    void initLumi(gen::StreamCache<HAD, DEC>* cache, LuminosityBlock const& index, EventSetup const& es) const;
    ParameterSet config_;
    InputTag runInfoProductTag_;
    EDGetTokenT<LHERunInfoProduct> runInfoProductToken_;
    EDGetTokenT<LHEEventProduct> eventProductToken_;
    unsigned int counterRunInfoProducts_;
    unsigned int nAttempts_;
    mutable std::atomic<gen::StreamCache<HAD, DEC>*> useInLumi_{nullptr};
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

    //sum of weights for events passing hadronization
    double waccept = 0;
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

      waccept += genEventInfo->weight();
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
  void ConcurrentHadronizerFilter<HAD, DEC>::streamBeginRun(StreamID id, Run const& run, EventSetup const& es) const {
    // this is run-specific

    // get LHE stuff and pass to hadronizer!

    if (counterRunInfoProducts_ > 1)
      throw edm::Exception(errors::EventCorruption) << "More than one LHERunInfoProduct present";

    if (counterRunInfoProducts_ == 0)
      throw edm::Exception(errors::EventCorruption) << "No LHERunInfoProduct present";

    edm::Handle<LHERunInfoProduct> lheRunInfoProduct;
    run.getByLabel(runInfoProductTag_, lheRunInfoProduct);
    //TODO: fix so that this actually works with getByToken commented below...
    //run.getByToken(runInfoProductToken_, lheRunInfoProduct);
    auto& hadronizer = this->streamCache(id)->hadronizer_;

    hadronizer.setLHERunInfo(std::make_unique<lhef::LHERunInfo>(*lheRunInfoProduct));
    lhef::LHERunInfo* lheRunInfo = hadronizer.getLHERunInfo().get();
    lheRunInfo->initLumi();
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
  }

  template <class HAD, class DEC>
  std::shared_ptr<gen::RunCache> ConcurrentHadronizerFilter<HAD, DEC>::globalBeginRun(edm::Run const&,
                                                                                      edm::EventSetup const&) const {
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
    if (useInLumi_ != this->streamCache(id)) {
      initLumi(this->streamCache(id), lumi, es);
    } else {
      useInLumi_.store(nullptr);
    }
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
    cache->decayer_->setRandomEngine(enginePtr.get());

    auto unsetH = [](HAD* h) { h->setRandomEngine(nullptr); };
    auto unsetD = [](DEC* d) { d->setRandomEngine(nullptr); };

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
  void ConcurrentHadronizerFilter<HAD, DEC>::globalBeginLuminosityBlockProduce(LuminosityBlock& lumi,
                                                                               EventSetup const& es) const {
    //need one of the streams to finish
    while (useInLumi_.load() == nullptr) {
    }
    initLumi(useInLumi_, lumi, es);
    std::unique_ptr<GenLumiInfoHeader> genLumiInfoHeader(useInLumi_.load()->hadronizer_.getGenLumiInfoHeader());
    lumi.put(std::move(genLumiInfoHeader));
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

    gen::StreamCache<HAD, DEC>* expected = nullptr;
    //make it available for beginLuminosityBlockProduce
    useInLumi_.compare_exchange_strong(expected, this->streamCache(id));
  }

  template <class HAD, class DEC>
  std::shared_ptr<gen::LumiSummary> ConcurrentHadronizerFilter<HAD, DEC>::globalBeginLuminosityBlockSummary(
      edm::LuminosityBlock const&, edm::EventSetup const&) const {
    return std::make_shared<gen::LumiSummary>();
  }

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
