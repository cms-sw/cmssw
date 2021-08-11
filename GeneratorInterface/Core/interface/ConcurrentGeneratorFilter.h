// -*- C++ -*-
//
//

// class template ConcurrentGeneratorFilter<HAD> provides an EDFilter which uses
// the hadronizer type HAD to generate partons, hadronize them,
// and decay the resulting particles, in the CMS framework.

#ifndef GeneratorInterface_Core_ConcurrentGeneratorFilter_h
#define GeneratorInterface_Core_ConcurrentGeneratorFilter_h

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
    struct GenRunCache {
      mutable std::atomic<GenRunInfoProduct*> product_{nullptr};
      ~GenRunCache() { delete product_.load(); }

      // This is called from globalEndRunProduce which is known to
      // be safe as the framework would not be calling any other
      // methods of this module using this run at that time
      std::unique_ptr<GenRunInfoProduct> release() const noexcept {
        auto retValue = product_.load();
        product_.store(nullptr);
        return std::unique_ptr<GenRunInfoProduct>(retValue);
      }
    };
    struct GenLumiSummary {
      mutable std::unique_ptr<GenLumiInfoProduct> lumiInfo_;
    };
    template <typename HAD, typename DEC>
    struct GenStreamCache {
      GenStreamCache(ParameterSet const& iPSet) : hadronizer_{iPSet}, nEventsInLumiBlock_{0} {}
      HAD hadronizer_;
      std::unique_ptr<DEC> decayer_;
      unsigned int nEventsInLumiBlock_;
      bool initialized_ = false;
    };
  }  // namespace gen

  template <class HAD, class DEC>
  class ConcurrentGeneratorFilter : public global::EDFilter<EndRunProducer,
                                                            BeginLuminosityBlockProducer,
                                                            EndLuminosityBlockProducer,
                                                            RunCache<gen::GenRunCache>,
                                                            LuminosityBlockSummaryCache<gen::GenLumiSummary>,
                                                            StreamCache<gen::GenStreamCache<HAD, DEC>>> {
  public:
    typedef HAD Hadronizer;
    typedef DEC Decayer;

    // The given ParameterSet will be passed to the contained Hadronizer object.
    explicit ConcurrentGeneratorFilter(ParameterSet const& ps);

    bool filter(StreamID id, Event& e, EventSetup const& es) const override;
    std::unique_ptr<gen::GenStreamCache<HAD, DEC>> beginStream(StreamID) const override;
    std::shared_ptr<gen::GenRunCache> globalBeginRun(Run const&, EventSetup const&) const override;
    std::shared_ptr<gen::GenLumiSummary> globalBeginLuminosityBlockSummary(LuminosityBlock const&,
                                                                           EventSetup const&) const override;
    void globalBeginLuminosityBlockProduce(LuminosityBlock&, EventSetup const&) const override;
    void streamBeginLuminosityBlock(StreamID, LuminosityBlock const&, EventSetup const&) const override;
    void streamEndLuminosityBlock(StreamID, LuminosityBlock const&, EventSetup const&) const override;
    void streamEndLuminosityBlockSummary(StreamID,
                                         LuminosityBlock const&,
                                         EventSetup const&,
                                         gen::GenLumiSummary*) const override;
    void globalEndLuminosityBlockSummary(LuminosityBlock const&,
                                         EventSetup const&,
                                         gen::GenLumiSummary*) const override;
    void globalEndLuminosityBlockProduce(LuminosityBlock&,
                                         EventSetup const&,
                                         gen::GenLumiSummary const*) const override;
    void streamEndRun(StreamID, Run const&, EventSetup const&) const override;
    void globalEndRun(Run const&, EventSetup const&) const override;
    void globalEndRunProduce(Run&, EventSetup const&) const override;

  private:
    void initLumi(gen::GenStreamCache<HAD, DEC>* cache, LuminosityBlock const& index, EventSetup const& es) const;
    ParameterSet config_;
    mutable std::atomic<gen::GenStreamCache<HAD, DEC>*> useInLumi_{nullptr};
  };

  //------------------------------------------------------------------------
  //
  // Implementation

  template <class HAD, class DEC>
  ConcurrentGeneratorFilter<HAD, DEC>::ConcurrentGeneratorFilter(ParameterSet const& ps) : config_(ps) {
    // TODO:
    // Put the list of types produced by the filters here.
    // The current design calls for:
    //   * LHEGeneratorInfo
    //   * LHEEvent
    //   * HepMCProduct
    // But I can not find the LHEGeneratorInfo class; it might need to
    // be invented.

    this->template produces<HepMCProduct>("unsmeared");
    this->template produces<GenEventInfoProduct>();
    this->template produces<GenLumiInfoHeader, edm::Transition::BeginLuminosityBlock>();
    this->template produces<GenLumiInfoProduct, edm::Transition::EndLuminosityBlock>();
    this->template produces<GenRunInfoProduct, edm::Transition::EndRun>();
  }

  template <class HAD, class DEC>
  std::unique_ptr<gen::GenStreamCache<HAD, DEC>> ConcurrentGeneratorFilter<HAD, DEC>::beginStream(StreamID) const {
    auto cache = std::make_unique<gen::GenStreamCache<HAD, DEC>>(config_);

    if (config_.exists("ExternalDecays")) {
      ParameterSet ps1 = config_.getParameter<ParameterSet>("ExternalDecays");
      cache->decayer_.reset(new Decayer(ps1));
    }

    // We need a hadronizer during globalBeginLumiProduce, doesn't matter which one
    gen::GenStreamCache<HAD, DEC>* expected = nullptr;
    useInLumi_.compare_exchange_strong(expected, cache.get());

    return cache;
  }

  template <class HAD, class DEC>
  std::shared_ptr<gen::GenRunCache> ConcurrentGeneratorFilter<HAD, DEC>::globalBeginRun(Run const&,
                                                                                        EventSetup const&) const {
    return std::make_shared<gen::GenRunCache>();
  }

  template <class HAD, class DEC>
  std::shared_ptr<gen::GenLumiSummary> ConcurrentGeneratorFilter<HAD, DEC>::globalBeginLuminosityBlockSummary(
      LuminosityBlock const&, EventSetup const&) const {
    return std::make_shared<gen::GenLumiSummary>();
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::initLumi(gen::GenStreamCache<HAD, DEC>* cache,
                                                     LuminosityBlock const& lumi,
                                                     EventSetup const& es) const {
    cache->nEventsInLumiBlock_ = 0;

    // We need all copies to see same random # for begin lumi
    Service<RandomNumberGenerator> rng;
    auto enginePtr = rng->cloneEngine(lumi.index());
    cache->hadronizer_.setRandomEngine(enginePtr.get());
    cache->decayer_->setRandomEngine(enginePtr.get());

    auto unsetH = [](HAD* h) { h->setRandomEngine(nullptr); };
    auto unsetD = [](DEC* d) { d->setRandomEngine(nullptr); };

    std::unique_ptr<HAD, decltype(unsetH)> randomEngineSentry(&cache->hadronizer_, unsetH);
    std::unique_ptr<DEC, decltype(unsetD)> randomEngineSentryDecay(cache->decayer_.get(), unsetD);

    cache->hadronizer_.randomizeIndex(lumi, enginePtr.get());

    if (!cache->hadronizer_.readSettings(0))
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

    if (!cache->hadronizer_.initializeForInternalPartons())
      throw edm::Exception(errors::Configuration)
          << "Failed to initialize hadronizer " << cache->hadronizer_.classname()
          << " for internal parton generation\n";

    cache->initialized_ = true;
  }

  template <class HAD, class DEC>
  bool ConcurrentGeneratorFilter<HAD, DEC>::filter(StreamID id, Event& ev, EventSetup const& /* es */) const {
    auto cache = this->streamCache(id);
    RandomEngineSentry<HAD> randomEngineSentry(&cache->hadronizer_, ev.streamID());
    RandomEngineSentry<DEC> randomEngineSentryDecay(cache->decayer_.get(), ev.streamID());

    cache->hadronizer_.setEDMEvent(ev);

    bool passEvtGenSelector = false;
    std::unique_ptr<HepMC::GenEvent> event(nullptr);

    while (!passEvtGenSelector) {
      event.reset();
      cache->hadronizer_.setEDMEvent(ev);

      if (!cache->hadronizer_.generatePartonsAndHadronize())
        return false;

      // this is "fake" stuff
      // in principle, decays are done as part of full event generation,
      // except for particles that are marked as to be kept stable
      // but we currently keep in it the design, because we might want
      // to use such feature for other applications
      //
      if (!cache->hadronizer_.decay())
        return false;

      event = cache->hadronizer_.getGenEvent();
      if (!event.get())
        return false;

      //
      // The external decay driver is being added to the system, it should be called here
      //
      if (cache->decayer_) {
        auto t = cache->decayer_->decay(event.get());
        if (t != event.get()) {
          event.reset(t);
        }
      }
      if (!event.get())
        return false;

      passEvtGenSelector = cache->hadronizer_.select(event.get());
    }

    // check and perform if there're any unstable particles after
    // running external decay packages
    //
    // fisrt of all, put back modified event tree (after external decay)
    //
    cache->hadronizer_.resetEvent(std::move(event));

    //
    // now run residual decays
    //
    if (!cache->hadronizer_.residualDecay())
      return false;

    cache->hadronizer_.finalizeEvent();

    event = cache->hadronizer_.getGenEvent();
    if (!event.get())
      return false;

    event->set_event_number(ev.id().event());

    //
    // finally, form up EDM products !
    //
    std::unique_ptr<GenEventInfoProduct> genEventInfo(cache->hadronizer_.getGenEventInfo());
    if (!genEventInfo.get()) {
      // create GenEventInfoProduct from HepMC event in case hadronizer didn't provide one
      genEventInfo = std::make_unique<GenEventInfoProduct>(event.get());
    }

    ev.put(std::move(genEventInfo));

    std::unique_ptr<HepMCProduct> bare_product(new HepMCProduct());
    bare_product->addHepMCData(event.release());
    ev.put(std::move(bare_product), "unsmeared");
    cache->nEventsInLumiBlock_++;
    return true;
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::globalBeginLuminosityBlockProduce(LuminosityBlock& lumi,
                                                                              EventSetup const& es) const {
    // need one of the streams to finish
    while (useInLumi_.load() == nullptr) {
    }
    initLumi(useInLumi_, lumi, es);
    std::unique_ptr<GenLumiInfoHeader> genLumiInfoHeader(useInLumi_.load()->hadronizer_.getGenLumiInfoHeader());
    lumi.put(std::move(genLumiInfoHeader));
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::streamBeginLuminosityBlock(StreamID id,
                                                                       LuminosityBlock const& lumi,
                                                                       EventSetup const& es) const {
    if (useInLumi_ != this->streamCache(id)) {
      initLumi(this->streamCache(id), lumi, es);
    } else {
      useInLumi_.store(nullptr);
    }
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::streamEndLuminosityBlock(StreamID id,
                                                                     LuminosityBlock const&,
                                                                     EventSetup const&) const {
    this->streamCache(id)->hadronizer_.cleanLHE();
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::streamEndLuminosityBlockSummary(StreamID id,
                                                                            LuminosityBlock const&,
                                                                            EventSetup const&,
                                                                            gen::GenLumiSummary* iSummary) const {
    auto cache = this->streamCache(id);
    cache->hadronizer_.statistics();
    if (cache->decayer_)
      cache->decayer_->statistics();

    GenRunInfoProduct genRunInfo = GenRunInfoProduct(cache->hadronizer_.getGenRunInfo());
    std::vector<GenLumiInfoProduct::ProcessInfo> GenLumiProcess;
    const GenRunInfoProduct::XSec& xsec = genRunInfo.internalXSec();
    GenLumiInfoProduct::ProcessInfo temp;
    unsigned int nEvtInLumiBlock_ = cache->nEventsInLumiBlock_;
    temp.setProcess(0);
    temp.setLheXSec(xsec.value(), xsec.error());  // Pythia gives error of -1
    temp.setNPassPos(nEvtInLumiBlock_);
    temp.setNPassNeg(0);
    temp.setNTotalPos(nEvtInLumiBlock_);
    temp.setNTotalNeg(0);
    temp.setTried(nEvtInLumiBlock_, nEvtInLumiBlock_, nEvtInLumiBlock_);
    temp.setSelected(nEvtInLumiBlock_, nEvtInLumiBlock_, nEvtInLumiBlock_);
    temp.setKilled(nEvtInLumiBlock_, nEvtInLumiBlock_, nEvtInLumiBlock_);
    temp.setAccepted(0, -1, -1);
    temp.setAcceptedBr(0, -1, -1);
    GenLumiProcess.push_back(temp);

    GenLumiInfoProduct genLumiInfo;
    genLumiInfo.setHEPIDWTUP(-1);
    genLumiInfo.setProcessInfo(GenLumiProcess);

    if (iSummary->lumiInfo_) {
      iSummary->lumiInfo_->mergeProduct(genLumiInfo);
    } else {
      iSummary->lumiInfo_ = std::make_unique<GenLumiInfoProduct>(std::move(genLumiInfo));
    }

    cache->nEventsInLumiBlock_ = 0;

    gen::GenStreamCache<HAD, DEC>* expected = nullptr;
    //make it available for beginLuminosityBlockProduce
    useInLumi_.compare_exchange_strong(expected, this->streamCache(id));
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::globalEndLuminosityBlockSummary(LuminosityBlock const&,
                                                                            EventSetup const&,
                                                                            gen::GenLumiSummary*) const {}

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::globalEndLuminosityBlockProduce(LuminosityBlock& lumi,
                                                                            EventSetup const&,
                                                                            gen::GenLumiSummary const* iSummary) const {
    lumi.put(std::move(iSummary->lumiInfo_));
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::streamEndRun(StreamID id, Run const& run, EventSetup const&) const {
    auto rCache = this->runCache(run.index());
    auto cache = this->streamCache(id);

    // If relevant, record the integrated luminosity for this run
    // here.  To do so, we would need a standard function to invoke on
    // the contained hadronizer that would report the integrated
    // luminosity.

    if (cache->initialized_) {
      cache->hadronizer_.statistics();
      if (cache->decayer_)
        cache->decayer_->statistics();
    }
    GenRunInfoProduct& genRunInfo = cache->hadronizer_.getGenRunInfo();
    GenRunInfoProduct* expect = nullptr;

    std::unique_ptr<GenRunInfoProduct> griproduct(new GenRunInfoProduct(genRunInfo));
    // All the GenRunInfoProducts for all streams shoule be identical, therefore we only need one
    if (rCache->product_.compare_exchange_strong(expect, griproduct.get())) {
      griproduct.release();
    }
  }

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::globalEndRun(Run const&, EventSetup const&) const {}

  template <class HAD, class DEC>
  void ConcurrentGeneratorFilter<HAD, DEC>::globalEndRunProduce(Run& run, EventSetup const&) const {
    run.put(this->runCache(run.index())->release());
  }

}  // namespace edm

#endif  // GeneratorInterface_Core_ConcurrentGeneratorFilter_h
