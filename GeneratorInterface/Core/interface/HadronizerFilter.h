// -*- C++ -*-
//
//

// class template HadronizerFilter<HAD> provides an EDFilter which uses
// the hadronizer type HAD to read in external partons and hadronize them,
// and decay the resulting particles, in the CMS framework.

#ifndef gen_HadronizerFilter_h
#define gen_HadronizerFilter_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "CLHEP/Random/RandomEngine.h"

// #include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Core/interface/HepMCFilterDriver.h"
#include "GeneratorInterface/Core/interface/HepMC3FilterDriver.h"

// LHE Run
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct3.h"

namespace edm {
  template <class HAD, class DEC>
  class HadronizerFilter : public one::EDFilter<EndRunProducer,
                                                BeginLuminosityBlockProducer,
                                                EndLuminosityBlockProducer,
                                                one::WatchRuns,
                                                one::WatchLuminosityBlocks,
                                                one::SharedResources> {
  public:
    typedef HAD Hadronizer;
    typedef DEC Decayer;

    // The given ParameterSet will be passed to the contained
    // Hadronizer object.
    explicit HadronizerFilter(ParameterSet const& ps);

    ~HadronizerFilter() override;

    bool filter(Event& e, EventSetup const& es) override;
    void beginRun(Run const&, EventSetup const&) override;
    void endRun(Run const&, EventSetup const&) override;
    void endRunProduce(Run&, EventSetup const&) override;
    void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
    void beginLuminosityBlockProduce(LuminosityBlock&, EventSetup const&) override;
    void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
    void endLuminosityBlockProduce(LuminosityBlock&, EventSetup const&) override;

  private:
    Hadronizer hadronizer_;
    // gen::ExternalDecayDriver* decayer_;
    Decayer* decayer_;
    HepMCFilterDriver* filter_;
    HepMC3FilterDriver* filter3_;
    InputTag runInfoProductTag_;
    EDGetTokenT<LHERunInfoProduct> runInfoProductToken_;
    EDGetTokenT<LHEEventProduct> eventProductToken_;
    unsigned int counterRunInfoProducts_;
    unsigned int nAttempts_;
    unsigned int ivhepmc = 2;
  };

  //------------------------------------------------------------------------
  //
  // Implementation

  template <class HAD, class DEC>
  HadronizerFilter<HAD, DEC>::HadronizerFilter(ParameterSet const& ps)
      : EDFilter(),
        hadronizer_(ps),
        decayer_(nullptr),
        filter_(nullptr),
        filter3_(nullptr),
        runInfoProductTag_(),
        runInfoProductToken_(),
        eventProductToken_(),
        counterRunInfoProducts_(0),
        nAttempts_(1) {
    callWhenNewProductsRegistered([this](ProductDescription const& iBD) {
      //this is called each time a module registers that it will produce a LHERunInfoProduct
      if (iBD.unwrappedTypeID() == edm::TypeID(typeid(LHERunInfoProduct)) && iBD.branchType() == InRun) {
        ++(this->counterRunInfoProducts_);
        this->eventProductToken_ = consumes<LHEEventProduct>(
            InputTag((iBD.moduleLabel() == "externalLHEProducer") ? "externalLHEProducer" : "source"));
        this->runInfoProductTag_ = InputTag(iBD.moduleLabel(), iBD.productInstanceName(), iBD.processName());
        this->runInfoProductToken_ = consumes<LHERunInfoProduct, InRun>(
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

    std::vector<std::string> const& sharedResources = hadronizer_.sharedResources();
    for (auto const& resource : sharedResources) {
      usesResource(resource);
    }

    if (ps.exists("ExternalDecays")) {
      //decayer_ = new gen::ExternalDecayDriver(ps.getParameter<ParameterSet>("ExternalDecays"));
      ParameterSet ps1 = ps.getParameter<ParameterSet>("ExternalDecays");
      decayer_ = new Decayer(ps1, consumesCollector());

      std::vector<std::string> const& sharedResourcesDec = decayer_->sharedResources();
      for (auto const& resource : sharedResourcesDec) {
        usesResource(resource);
      }
    }

    ivhepmc = hadronizer_.getVHepMC();
    if (ps.exists("HepMCFilter")) {
      ParameterSet psfilter = ps.getParameter<ParameterSet>("HepMCFilter");
      if (ivhepmc == 2) {
        filter_ = new HepMCFilterDriver(psfilter);
      } else if (ivhepmc == 3) {
        filter3_ = new HepMC3FilterDriver(psfilter);
      }
    }

    //initialize setting for multiple hadronization attempts
    if (ps.exists("nAttempts")) {
      nAttempts_ = ps.getParameter<unsigned int>("nAttempts");
    }

    // This handles the case where there are no shared resources, because you
    // have to declare something when the SharedResources template parameter was used.
    if (sharedResources.empty() && (!decayer_ || decayer_->sharedResources().empty())) {
      usesResource(edm::uniqueSharedResourceName());
    }

    if (ivhepmc == 2) {
      produces<edm::HepMCProduct>("unsmeared");
      produces<GenEventInfoProduct>();
    } else if (ivhepmc == 3) {
      produces<edm::HepMC3Product>("unsmeared");
      produces<GenEventInfoProduct3>();
    }
    produces<GenLumiInfoHeader, edm::Transition::BeginLuminosityBlock>();
    produces<GenLumiInfoProduct, edm::Transition::EndLuminosityBlock>();
    produces<GenRunInfoProduct, edm::Transition::EndRun>();
    if (filter_ || filter3_)
      produces<GenFilterInfo, edm::Transition::EndLuminosityBlock>();
  }

  template <class HAD, class DEC>
  HadronizerFilter<HAD, DEC>::~HadronizerFilter() {
    if (decayer_)
      delete decayer_;
    if (filter_)
      delete filter_;
    if (filter3_)
      delete filter3_;
  }

  template <class HAD, class DEC>
  bool HadronizerFilter<HAD, DEC>::filter(Event& ev, EventSetup const& /* es */) {
    RandomEngineSentry<HAD> randomEngineSentry(&hadronizer_, ev.streamID());
    RandomEngineSentry<DEC> randomEngineSentryDecay(decayer_, ev.streamID());

    hadronizer_.setEDMEvent(ev);

    // get LHE stuff and pass to hadronizer!
    //
    edm::Handle<LHEEventProduct> product;
    ev.getByToken(eventProductToken_, product);

    std::unique_ptr<HepMC::GenEvent> finalEvent;
    std::unique_ptr<HepMC3::GenEvent> finalEvent3;
    std::unique_ptr<GenEventInfoProduct> finalGenEventInfo;
    std::unique_ptr<GenEventInfoProduct3> finalGenEventInfo3;

    //number of accepted events
    unsigned int naccept = 0;

    for (unsigned int itry = 0; itry < nAttempts_; ++itry) {
      hadronizer_.setLHEEvent(std::make_unique<lhef::LHEEvent>(hadronizer_.getLHERunInfo(), *product));

      // hadronizer_.generatePartons();
      if (!hadronizer_.hadronize())
        continue;

      //  this is "fake" stuff
      // in principle, decays are done as part of full event generation,
      // except for particles that are marked as to be kept stable
      // but we currently keep in it the design, because we might want
      // to use such feature for other applications
      //
      if (!hadronizer_.decay())
        continue;

      std::unique_ptr<HepMC::GenEvent> event(hadronizer_.getGenEvent());
      std::unique_ptr<HepMC3::GenEvent> event3(hadronizer_.getGenEvent3());
      if (ivhepmc == 2 && !event.get())
        continue;
      if (ivhepmc == 3 && !event3.get())
        continue;

      // The external decay driver is being added to the system,
      // it should be called here
      //
      if (decayer_) {
        auto lheEvent = hadronizer_.getLHEEvent();
        auto t = decayer_->decay(event.get(), lheEvent.get());
        if (t != event.get()) {
          event.reset(t);
        }
        hadronizer_.setLHEEvent(std::move(lheEvent));
      }

      if (ivhepmc == 2 && !event.get())
        continue;
      if (ivhepmc == 3 && !event3.get())
        continue;

      // check and perform if there're any unstable particles after
      // running external decay packges
      //
      hadronizer_.resetEvent(std::move(event));
      hadronizer_.resetEvent3(std::move(event3));
      if (!hadronizer_.residualDecay())
        continue;

      hadronizer_.finalizeEvent();

      event = hadronizer_.getGenEvent();
      event3 = hadronizer_.getGenEvent3();
      if (ivhepmc == 2 && !event.get())
        continue;
      if (ivhepmc == 3 && !event3.get())
        continue;

      if (ivhepmc == 2) {  // HepMC
        event->set_event_number(ev.id().event());
        std::unique_ptr<GenEventInfoProduct> genEventInfo(hadronizer_.getGenEventInfo());
        if (!genEventInfo.get()) {
          // create GenEventInfoProduct from HepMC event in case hadronizer didn't provide one
          genEventInfo = std::make_unique<GenEventInfoProduct>(event.get());
        }

        //if HepMCFilter was specified, test event
        if (filter_ && !filter_->filter(event.get(), genEventInfo->weight()))
          continue;

        ++naccept;

        //keep the LAST accepted event (which is equivalent to choosing randomly from the accepted events)
        finalEvent = std::move(event);
        finalGenEventInfo = std::move(genEventInfo);
      } else if (ivhepmc == 3) {  // HepMC3
        event3->set_event_number(ev.id().event());
        std::unique_ptr<GenEventInfoProduct3> genEventInfo3(hadronizer_.getGenEventInfo3());
        if (!genEventInfo3.get()) {
          // create GenEventInfoProduct3 from HepMC3 event in case hadronizer didn't provide one
          genEventInfo3 = std::make_unique<GenEventInfoProduct3>(event3.get());
        }

        //if HepMCFilter was specified, test event
        if (filter3_ && !filter3_->filter(event3.get(), genEventInfo3->weight()))
          continue;

        ++naccept;

        //keep the LAST accepted event (which is equivalent to choosing randomly from the accepted events)
        finalEvent3 = std::move(event3);
        finalGenEventInfo3 = std::move(genEventInfo3);
      }
    }

    if (!naccept)
      return false;

    if (ivhepmc == 2) {  // HepMC
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
    } else if (ivhepmc == 3) {  // HepMC3
      //adjust event weights if necessary (in case input event was attempted multiple times)
      if (nAttempts_ > 1) {
        double multihadweight = double(naccept) / double(nAttempts_);

        //adjust weight for GenEventInfoProduct
        finalGenEventInfo3->weights()[0] *= multihadweight;

        //adjust weight for HepMC GenEvent (used e.g for RIVET)
        finalEvent3->weights()[0] *= multihadweight;
      }

      ev.put(std::move(finalGenEventInfo3));

      std::unique_ptr<HepMC3Product> bare_product(new HepMC3Product());
      bare_product->addHepMCData(finalEvent3.release());
      ev.put(std::move(bare_product), "unsmeared");
    }

    return true;
  }

  template <class HAD, class DEC>
  void HadronizerFilter<HAD, DEC>::beginRun(Run const& run, EventSetup const& es) {
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

    hadronizer_.setLHERunInfo(std::make_unique<lhef::LHERunInfo>(*lheRunInfoProduct));
    lhef::LHERunInfo* lheRunInfo = hadronizer_.getLHERunInfo().get();
    lheRunInfo->initLumi();
  }

  template <class HAD, class DEC>
  void HadronizerFilter<HAD, DEC>::endRun(Run const& r, EventSetup const&) {}

  template <class HAD, class DEC>
  void HadronizerFilter<HAD, DEC>::endRunProduce(Run& r, EventSetup const&) {
    // Retrieve the LHE run info summary and transfer determined
    // cross-section into the generator run info

    const lhef::LHERunInfo* lheRunInfo = hadronizer_.getLHERunInfo().get();
    lhef::LHERunInfo::XSec xsec = lheRunInfo->xsec();

    GenRunInfoProduct& genRunInfo = hadronizer_.getGenRunInfo();
    genRunInfo.setInternalXSec(GenRunInfoProduct::XSec(xsec.value(), xsec.error()));

    // If relevant, record the integrated luminosity for this run
    // here.  To do so, we would need a standard function to invoke on
    // the contained hadronizer that would report the integrated
    // luminosity.

    hadronizer_.statistics();
    if (decayer_)
      decayer_->statistics();
    if (filter_)
      filter_->statistics();
    if (filter3_)
      filter3_->statistics();
    lheRunInfo->statistics();

    std::unique_ptr<GenRunInfoProduct> griproduct(new GenRunInfoProduct(genRunInfo));
    r.put(std::move(griproduct));
  }

  template <class HAD, class DEC>
  void HadronizerFilter<HAD, DEC>::beginLuminosityBlock(LuminosityBlock const& lumi, EventSetup const& es) {}

  template <class HAD, class DEC>
  void HadronizerFilter<HAD, DEC>::beginLuminosityBlockProduce(LuminosityBlock& lumi, EventSetup const& es) {
    lhef::LHERunInfo* lheRunInfo = hadronizer_.getLHERunInfo().get();
    lheRunInfo->initLumi();

    RandomEngineSentry<HAD> randomEngineSentry(&hadronizer_, lumi.index());
    RandomEngineSentry<DEC> randomEngineSentryDecay(decayer_, lumi.index());

    hadronizer_.randomizeIndex(lumi, randomEngineSentry.randomEngine());

    if (!hadronizer_.readSettings(1))
      throw edm::Exception(errors::Configuration)
          << "Failed to read settings for the hadronizer " << hadronizer_.classname() << " \n";

    if (decayer_) {
      decayer_->init(es);
      if (!hadronizer_.declareStableParticles(decayer_->operatesOnParticles()))
        throw edm::Exception(errors::Configuration) << "Failed to declare stable particles in hadronizer "
                                                    << hadronizer_.classname() << " for internal parton generation\n";
      if (!hadronizer_.declareSpecialSettings(decayer_->specialSettings()))
        throw edm::Exception(errors::Configuration)
            << "Failed to declare special settings in hadronizer " << hadronizer_.classname() << "\n";
    }

    if (filter_) {
      filter_->resetStatistics();
    }
    if (filter3_) {
      filter3_->resetStatistics();
    }

    if (!hadronizer_.initializeForExternalPartons())
      throw edm::Exception(errors::Configuration)
          << "Failed to initialize hadronizer " << hadronizer_.classname() << " for external parton generation\n";

    std::unique_ptr<GenLumiInfoHeader> genLumiInfoHeader(hadronizer_.getGenLumiInfoHeader());
    lumi.put(std::move(genLumiInfoHeader));
  }

  template <class HAD, class DEC>
  void HadronizerFilter<HAD, DEC>::endLuminosityBlock(LuminosityBlock const&, EventSetup const&) {}

  template <class HAD, class DEC>
  void HadronizerFilter<HAD, DEC>::endLuminosityBlockProduce(LuminosityBlock& lumi, EventSetup const&) {
    const lhef::LHERunInfo* lheRunInfo = hadronizer_.getLHERunInfo().get();

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
    std::unique_ptr<GenLumiInfoProduct> genLumiInfo(new GenLumiInfoProduct());
    genLumiInfo->setHEPIDWTUP(lheRunInfo->getHEPRUP()->IDWTUP);
    genLumiInfo->setProcessInfo(GenLumiProcess);

    lumi.put(std::move(genLumiInfo));

    // produce GenFilterInfo if HepMCFilter is called
    if (filter_) {
      std::unique_ptr<GenFilterInfo> thisProduct(new GenFilterInfo(filter_->numEventsPassPos(),
                                                                   filter_->numEventsPassNeg(),
                                                                   filter_->numEventsTotalPos(),
                                                                   filter_->numEventsTotalNeg(),
                                                                   filter_->sumpass_w(),
                                                                   filter_->sumpass_w2(),
                                                                   filter_->sumtotal_w(),
                                                                   filter_->sumtotal_w2()));
      lumi.put(std::move(thisProduct));
    }
    if (filter3_) {
      std::unique_ptr<GenFilterInfo> thisProduct(new GenFilterInfo(filter3_->numEventsPassPos(),
                                                                   filter3_->numEventsPassNeg(),
                                                                   filter3_->numEventsTotalPos(),
                                                                   filter3_->numEventsTotalNeg(),
                                                                   filter3_->sumpass_w(),
                                                                   filter3_->sumpass_w2(),
                                                                   filter3_->sumtotal_w(),
                                                                   filter3_->sumtotal_w2()));
      lumi.put(std::move(thisProduct));
    }
  }

}  // namespace edm

#endif  // gen_HadronizerFilter_h
